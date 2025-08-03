import torch
from torch.optim.optimizer import Optimizer
import logging
import math

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdaBeliefBF(Optimizer):
    """
    AdaBeliefBF - это реализация оптимизатора AdaBelief.
    
    AdaBelief адаптирует размер шага на основе "правдоподобности" градиента, измеряя разницу
    между текущим градиентом и его скользящим средним. Это может привести к более быстрой
    сходимости и лучшей обобщающей способности модели.

    Args:
        params (iterable): Итерируемый объект с параметрами для оптимизации или словари, определяющие группы параметров.
        lr (float, optional): Скорость обучения (по умолчанию: 1e-3).
        betas (Tuple[float, float], optional): Коэффициенты для вычисления скользящих средних градиента и его квадрата (по умолчанию: (0.9, 0.999)).
        eps (float, optional): Член, добавляемый к знаменателю для численной стабильности (по умолчанию: 1e-8).
        weight_decay (float, optional): Коэффициент затухания весов (L2-регуляризация) (по умолчанию: 0).
        rectify (bool, optional): Использовать ли механизм ректификации (RAdam) для стабилизации в начале обучения (по умолчанию: True).
        debug (bool, optional): Включить ли подробное логирование (по умолчанию: True).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, rectify=True, debug=True):
        if not 0.0 <= lr:
            raise ValueError(f"Некорректная скорость обучения: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Некорректное значение epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Некорректное значение beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Некорректное значение beta2: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, rectify=rectify)
        super(AdaBeliefBF, self).__init__(params, defaults)
        
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Выполняет один шаг оптимизации.

        Args:
            closure (callable, optional): Функция, которая пересчитывает модель и возвращает потери.

        Returns:
            loss: Значение потерь, если closure предоставлена, иначе None.
        """
        loss = None
        if closure is not not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            rectify = group['rectify']

            if self.debug:
                logger.debug(f"Группа параметров | Скорость обучения: {lr}, Затухание весов: {weight_decay}")

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdaBelief не поддерживает разреженные градиенты')

                state = self.state[p]

                # Инициализация состояния
                if len(state) == 0:
                    state['step'] = 0
                    # Первый момент (экспоненциальное скользящее среднее градиента)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Второй момент (экспоненциальное скользящее среднее квадрата разницы градиента)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1
                
                # Применение затухания весов (L2-регуляризация)
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                # Обновление первого момента (m_t)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # --- Ключевая логика AdaBelief ---
                # Вычисляем разницу между градиентом и его скользящим средним
                grad_residual = grad - exp_avg
                # Обновление второго момента (s_t)
                exp_avg_sq.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)
                # ---------------------------------

                # Коррекция смещения для первого и второго моментов
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Знаменатель для шага обновления
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                
                step_size = lr / bias_correction1

                # Механизм ректификации (из RAdam) для стабилизации
                if rectify:
                    rho_inf = 2 / (1 - beta2) - 1
                    rho_t = rho_inf - 2 * state['step'] * (beta2 ** state['step']) / (1 - beta2 ** state['step'])
                    
                    if rho_t > 5.0: # Пороговое значение, как в статье RAdam
                        rectified_step_size = math.sqrt(((rho_t - 4) * (rho_t - 2) * rho_inf) / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                        p.addcdiv_(exp_avg, denom, value=-step_size * rectified_step_size)
                    else:
                        p.add_(exp_avg, alpha=-step_size)
                else:
                    p.addcdiv_(exp_avg, denom, value=-step_size)

                # Логирование для отладки
                if self.debug:
                    param_norm = torch.norm(p).item()
                    grad_norm = torch.norm(p.grad).item() # Используем оригинальный градиент
                    exp_avg_norm = torch.norm(exp_avg).item()
                    exp_avg_sq_norm = torch.norm(exp_avg_sq).item()
                    logger.debug(
                        f"Параметр: норма={param_norm:.6f} | "
                        f"Градиент: норма={grad_norm:.6f} | "
                        f"Первый момент: норма={exp_avg_norm:.6f} | "
                        f"Второй момент: норма={exp_avg_sq_norm:.6f}"
                    )
        return loss
