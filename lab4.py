"""
Задача пошагового управления инвестиционным портфелем
Метод динамического программирования
Критерий Байеса
"""

from functools import lru_cache
from itertools import product

# ==========================================================
# 1. Исходные данные
# ==========================================================

# Начальное состояние (q1, q2, qD, cash)
INITIAL_STATE = (100.0, 800.0, 400.0, 600.0)

# Размеры пакетов
PACKET_CB1 = 25.0
PACKET_CB2 = 200.0
PACKET_DEP = 100.0

# Комиссии
GAMMA_CB1 = 0.04
GAMMA_CB2 = 0.07
GAMMA_DEP = 0.05

# Минимальные объёмы
MIN_CB1 = 30.0
MIN_CB2 = 150.0
MIN_DEP = 100.0

# ==========================================================
# 2. Вероятности и сценарии
# ==========================================================

# Вероятности сценариев
PROBS = {
    1: [0.6, 0.3, 0.1],
    2: [0.3, 0.2, 0.5],
    3: [0.4, 0.4, 0.2],
}

# Множители изменения стоимости активов
MULT = {
    1: [
        {"cb1": 1.20, "cb2": 1.10, "dep": 1.07},
        {"cb1": 1.05, "cb2": 1.02, "dep": 1.03},
        {"cb1": 0.80, "cb2": 0.95, "dep": 1.00},
    ],
    2: [
        {"cb1": 1.40, "cb2": 1.15, "dep": 1.01},
        {"cb1": 1.05, "cb2": 1.00, "dep": 1.00},
        {"cb1": 0.60, "cb2": 0.90, "dep": 1.00},
    ],
    3: [
        {"cb1": 1.15, "cb2": 1.12, "dep": 1.05},
        {"cb1": 1.05, "cb2": 1.01, "dep": 1.01},
        {"cb1": 0.70, "cb2": 0.94, "dep": 1.00},
    ],
}

# ==========================================================
# 3. Управления и политика
# ==========================================================

# Возможные управления (k1, k2, kD)
ACTIONS = list(product([-1, 0, 1], repeat=3))

# Оптимальная политика
POLICY = {}

# ==========================================================
# 4. Вспомогательные функции
# ==========================================================

def round_state(state, ndigits=2):
    """Округление состояния для использования в кэше."""
    return tuple(round(x, ndigits) for x in state)


# ==========================================================
# 5. Применение управления
# ==========================================================

def apply_control(state, action):
    """
    Применение управления к состоянию.
    Возвращает новое состояние или None при нарушении ограничений.
    """
    q1, q2, qD, cash = state
    k1, k2, kD = action

    dq1 = PACKET_CB1 * k1
    dq2 = PACKET_CB2 * k2
    dqD = PACKET_DEP * kD

    q1_new = q1 + dq1
    q2_new = q2 + dq2
    qD_new = qD + dqD

    # Проверка минимальных объёмов
    if q1_new < MIN_CB1 or q2_new < MIN_CB2 or qD_new < MIN_DEP:
        return None

    cash_new = cash

    # Учёт комиссий
    if dq1 > 0:
        cash_new -= dq1 * (1 + GAMMA_CB1)
    elif dq1 < 0:
        cash_new += (-dq1) * (1 - GAMMA_CB1)

    if dq2 > 0:
        cash_new -= dq2 * (1 + GAMMA_CB2)
    elif dq2 < 0:
        cash_new += (-dq2) * (1 - GAMMA_CB2)

    if dqD > 0:
        cash_new -= dqD * (1 + GAMMA_DEP)
    elif dqD < 0:
        cash_new += (-dqD) * (1 - GAMMA_DEP)

    # Запрет на кредитование
    if cash_new < 0:
        return None

    return (q1_new, q2_new, qD_new, cash_new)


# ==========================================================
# 6. Применение сценария
# ==========================================================

def apply_scenario(state, t, j):
    """Применение j-го сценария на этапе t."""
    q1, q2, qD, cash = state
    m = MULT[t][j]

    return (
        q1 * m["cb1"],
        q2 * m["cb2"],
        qD * m["dep"],
        cash
    )


# ==========================================================
# 7. Терминальная функция
# ==========================================================

def terminal_value(state):
    """Итоговая стоимость портфеля."""
    return sum(state)


# ==========================================================
# 8. Функция Беллмана
# ==========================================================

@lru_cache(maxsize=None)
def V(t, q1, q2, qD, cash, criterion="bayes"):
    state = round_state((q1, q2, qD, cash))

    # Конец горизонта
    if t == 4:
        return terminal_value(state)

    best_value = float("-inf")
    best_action = (0, 0, 0)

    for action in ACTIONS:
        after_ctrl = apply_control(state, action)
        if after_ctrl is None:
            continue

        expected = 0.0
        for j in range(3):
            expected += PROBS[t][j] * V(
                t + 1,
                *apply_scenario(after_ctrl, t, j),
                criterion
            )

        if expected > best_value:
            best_value = expected
            best_action = action

    POLICY[(t, state)] = best_action
    return best_value


# ==========================================================
# 9. Основной запуск
# ==========================================================

def main():
    POLICY.clear()
    V.cache_clear()

    criterion = "bayes"
    print("Критерий принятия решений:", criterion)
    print("Начальное состояние:", INITIAL_STATE)
    print()

    max_value = V(1, *INITIAL_STATE, criterion)
    print("Максимальное значение целевой функции V1:", round(max_value, 2))
    print()

    # Восстановление оптимального плана
    state = INITIAL_STATE
    print("Оптимальное управление по этапам:")

    for t in range(1, 4):
        sr = round_state(state)
        action = POLICY.get((t, sr), (0, 0, 0))

        print(f"Этап {t}:")
        print(f"  Состояние: {sr}")
        print(f"  Оптимальное управление (k1, k2, kD): {action}")

        after_ctrl = apply_control(sr, action)
        if after_ctrl is None:
            print("  Управление недопустимо.")
            break

        # Ожидаемое состояние
        exp_q1 = exp_q2 = exp_qD = 0.0
        cash_after = after_ctrl[3]

        for j in range(3):
            p = PROBS[t][j]
            ns = apply_scenario(after_ctrl, t, j)
            exp_q1 += p * ns[0]
            exp_q2 += p * ns[1]
            exp_qD += p * ns[2]

        state = (exp_q1, exp_q2, exp_qD, cash_after)
        print("  Ожидаемое состояние после этапа:", round_state(state))
        print()

    print("Вычисления завершены.")


if __name__ == "__main__":
    main()
