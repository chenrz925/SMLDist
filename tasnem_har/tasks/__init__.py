__all__ = [
    'TASNeMTeacherTrainTask',
    'TASNeMStudentTrainTask',
]

from collections import OrderedDict
from copy import deepcopy
from logging import Logger
from typing import List, Text

import torch
from ignite import engine, metrics
from ignite.engine import Events
from ignite.utils import convert_tensor
from tasker import Profile, Return, value, Definition
from tasker.storage import Storage
from tasker.tasks.torch import SimpleTrainTask
from torch import nn, optim
from torch.utils.data import DataLoader

from ..models.backbone import TASNeModel
from ..models.loss_functions import LogitsDistillingLoss


class TASNeMTeacherTrainTask(SimpleTrainTask):
    Model = TASNeModel

    def create_model(self, profile: Profile, shared: Storage, logger: Logger) -> nn.Module:
        if torch.cuda.is_available():
            return self.Model(**profile).cuda()
        else:
            return self.Model(**profile)

    @classmethod
    def define_model(cls):
        return [
            value('in_channels', int),
            value('out_channels', int),
            value('num_classes', int),
            value('layers', list, [
                [
                    value('kernel_size', int),
                    value('in_channels', int),
                    value('expansion', int),
                    value('out_channels', int),
                    value('attention', str),
                    value('activation', str),
                    value('stride', int),
                    value('anchor', bool)
                ]
            ]),
            value('simh', str, [
                value('update_steps_max', int),
                value('update_steps_eps', float),
                value('eval_mode', str)
            ])
        ]

    def require(self) -> List[Text]:
        require = list(super().require())
        require.extend(self.provide())
        return require

    def more_metrics(self, metrics_: OrderedDict):
        metrics_['loss'] = metrics.Loss(nn.CrossEntropyLoss())
        metrics_['accuracy'] = metrics.Accuracy()
        recall = metrics.Recall()
        precision = metrics.Precision()
        metrics_['f1macro'] = (2 * (recall * precision) / (recall + precision)).mean()

    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        train_loader = shared['train_loader']
        validate_loader = shared['validate_loader']

        model = self.create_model(profile.model, shared, logger)
        optimizer_return = self.create_optimizer(
            model, profile.optimizer, shared, logger)
        if isinstance(optimizer_return, tuple):
            optimizer = optimizer_return[0]
            scheduler = optimizer_return[1] if len(
                optimizer_return) > 1 else None
        else:
            optimizer = optimizer_return
            scheduler = None
        loss = self.create_loss(profile.loss, shared, logger)
        metrics_dict = OrderedDict()
        self.more_metrics(metrics_dict)

        def prepare_batch(batch, device=None, non_blocking=False):
            return self.prepare_batch(deepcopy(batch), device, non_blocking)

        trainer = self.create_trainer(
            model, optimizer, loss,
            device=torch.device(profile.device),
            non_blocking=profile.non_blocking,
            prepare_batch=prepare_batch,
        )

        evaluator = self.create_evaluator(
            model, metrics_dict,
            device=torch.device(profile.device),
            non_blocking=profile.non_blocking,
            prepare_batch=prepare_batch,
        )

        context = {'best_loss': +100.0, 'best_accuracy': 0.0}

        @evaluator.on(engine.Events.EPOCH_COMPLETED)
        def save_best_accuracy(engine_):
            if engine_.state.metrics['accuracy'] > context['best_accuracy']:
                context['best_loss'] = engine_.state.metrics['loss']
                context['best_accuracy'] = engine_.state.metrics['accuracy']
                shared[self.PROVIDE_KEY] = model
            logger.info(f'Best accuracy {context["best_accuracy"]}')

        @trainer.on(engine.Events.STARTED)
        def on_epoch_started(engine_):
            return self.on_epoch_started(engine_, profile, shared, logger)

        @trainer.on(engine.Events.COMPLETED)
        def on_completed(engine_):
            return self.on_completed(engine_, profile, shared, logger)

        @trainer.on(engine.Events.ITERATION_STARTED)
        def on_iteration_started(engine_):
            return self.on_iteration_started(engine_, profile, shared, logger)

        @trainer.on(engine.Events.ITERATION_COMPLETED)
        def on_iteration_completed(engine_):
            return self.on_iteration_completed(engine_, profile, shared, logger)

        @trainer.on(engine.Events.EPOCH_STARTED)
        def on_epoch_started(engine_):
            return self.on_epoch_started(engine_, profile, shared, logger)

        @trainer.on(engine.Events.EPOCH_COMPLETED)
        def on_epoch_completed(engine_):
            evaluator.run(validate_loader)
            if scheduler is not None:
                scheduler.step(engine_.state.output)
            return self.on_epoch_completed(engine_, evaluator.state.metrics, profile, shared, logger)

        trainer.run(
            train_loader,
            max_epochs=profile.max_epochs,
        )

        logger.info(f'Best accuracy {context["best_accuracy"]}')
        return Return.SUCCESS.value


class TASNeMStudentTrainTask(TASNeMTeacherTrainTask):
    Model = TASNeModel

    def __init__(self, prefix=None):
        super(TASNeMStudentTrainTask, self).__init__('student')

    def create_trainer(
                self, model, optimizer, loss_fn, device, non_blocking, prepare_batch,
                output_transform=lambda x, y, y_pred, loss: loss.item()
        ):
            if device:
                model.to(device)

            def _update(engine, batch):
                model.train()
                optimizer.zero_grad()
                x, y = prepare_batch(batch, device=device,
                                     non_blocking=non_blocking)
                y_pred = model(x)
                loss = loss_fn(x, y_pred, y)
                loss.backward()
                optimizer.step()
                return output_transform(x, y, y_pred, loss)

            return engine.Engine(_update)

    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        teacher_model = shared['teacher_model']
        train_loader: DataLoader = shared['train_loader']
        validate_loader: DataLoader = shared['validate_loader']

        model = self.create_model(profile.model, shared, logger)
        model.classifier.load_state_dict(teacher_model.classifier.state_dict())

        device = torch.device(profile.device)
        non_blocking = profile.non_blocking
        optimizer_return = self.create_optimizer(
            model, profile.optimizer, shared, logger)
        if isinstance(optimizer_return, tuple):
            optimizer = optimizer_return[0]
            scheduler = optimizer_return[1] if len(
                optimizer_return) > 1 else None
        else:
            optimizer = optimizer_return
            scheduler = None

        classifier_loss = LogitsDistillingLoss(
            teacher_model, hardness_factor=1 if 'hardness_factor' not in profile else profile.hardness_factor)
        logger.info(f'hardness factor {classifier_loss.hardness_factor}')

        for stage_index in range(teacher_model.num_stages):
            logger.info(f'STAGE {stage_index}')
            stage_optimizer = optim.Adam(model.parameters(), lr=profile.stage_lr if 'stage_lr' in profile else 1e-4)

            model.stage(stage_index)
            teacher_model.stage(stage_index)
            teacher_model.eval()

            def prepare_stage_batch(batch, device=None, non_blocking=False):
                x, y = deepcopy(batch)
                x, y = convert_tensor(
                    x, device=device, non_blocking=non_blocking
                ), convert_tensor(
                    y, device=device, non_blocking=non_blocking
                )
                with torch.no_grad():
                    return x, teacher_model(x)

            stage_trainer = engine.create_supervised_trainer(
                model=model,
                optimizer=stage_optimizer,
                loss_fn=nn.MSELoss(),
                device=device,
                non_blocking=non_blocking,
                prepare_batch=prepare_stage_batch
            )

            stage_evaluator = engine.create_supervised_evaluator(
                model=model,
                metrics={
                    'mse': metrics.Loss(nn.MSELoss()),
                    'huber': metrics.Loss(nn.SmoothL1Loss())
                },
                device=device,
                prepare_batch=prepare_stage_batch
            )

            @stage_trainer.on(Events.EPOCH_COMPLETED)
            def trainer_epoch_complete(engine_):
                stage_evaluator.run(
                    validate_loader, 1
                )
                self.on_epoch_completed(
                    engine_, stage_evaluator.state.metrics, profile, shared, logger)

            stage_trainer.run(
                train_loader,
                profile.stage_epochs if 'stage_epochs' in profile else 200
            )

        model.stage(model.num_stages)
        teacher_model.stage()

        metrics_dict = OrderedDict()
        self.more_metrics(metrics_dict)

        def prepare_batch(batch, device=None, non_blocking=False):
            return self.prepare_batch(deepcopy(batch), device, non_blocking)

        trainer = self.create_trainer(
            model, optimizer, classifier_loss,
            device=torch.device(profile.device),
            non_blocking=profile.non_blocking,
            prepare_batch=prepare_batch,
        )

        evaluator = self.create_evaluator(
            model, metrics_dict,
            device=torch.device(profile.device),
            non_blocking=profile.non_blocking,
            prepare_batch=prepare_batch,
        )


        context = {'best_accuracy': 0}

        @evaluator.on(engine.Events.EPOCH_COMPLETED)
        def save_best_accuracy(engine_):
            if engine_.state.metrics['accuracy'] > context['best_accuracy']:
                context['best_accuracy'] = engine_.state.metrics['accuracy']
                shared[self.PROVIDE_KEY] = deepcopy(model)
            logger.info(f'Best accuracy {context["best_accuracy"]}')

        @trainer.on(engine.Events.STARTED)
        def on_epoch_started(engine_):
            return self.on_epoch_started(engine_, profile, shared, logger)

        @trainer.on(engine.Events.COMPLETED)
        def on_completed(engine_):
            return self.on_completed(engine_, profile, shared, logger)

        @trainer.on(engine.Events.ITERATION_STARTED)
        def on_iteration_started(engine_):
            return self.on_iteration_started(engine_, profile, shared, logger)

        @trainer.on(engine.Events.ITERATION_COMPLETED)
        def on_iteration_completed(engine_):
            return self.on_iteration_completed(engine_, profile, shared, logger)

        @trainer.on(engine.Events.EPOCH_STARTED)
        def on_epoch_started(engine_):
            return self.on_epoch_started(engine_, profile, shared, logger)

        @trainer.on(engine.Events.EPOCH_COMPLETED)
        def on_epoch_completed(engine_):
            evaluator.run(validate_loader)
            if scheduler is not None:
                scheduler.step(engine_.state.output)
            return self.on_epoch_completed(engine_, evaluator.state.metrics, profile, shared, logger)

        trainer.run(
            train_loader,
            max_epochs=profile.max_epochs,
        )
        logger.info(f'Best accuracy {context["best_accuracy"]}')

        return Return.SUCCESS.value

    def require(self) -> List[Text]:
        return ['student_model', 'teacher_model', 'train_loader', 'validate_loader']

    def provide(self) -> List[Text]:
        return ['student_model']

    def remove(self) -> List[Text]:
        return []

    def define_model(cls):
        definition = super(TASNeMStudentTrainTask, cls).define_model()
        definition.extend([
            value('hardness_factor', float),
            value('stage_epochs', int),
            value('stage_lr', float),
        ])
        return definition

    def more_metrics(self, metrics_: OrderedDict):
        metrics_['loss'] = metrics.Loss(nn.CrossEntropyLoss())
        metrics_['accuracy'] = metrics.Accuracy()
        recall = metrics.Recall()
        precision = metrics.Precision()
        metrics_['f1macro'] = (2 * (recall * precision) / (recall + precision)).mean()
