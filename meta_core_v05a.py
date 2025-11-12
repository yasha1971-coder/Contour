# Mini-regulator v0.5 (exposed for reuse)
def self_goal_update(engine, metrics, k_noise=0.25, k_decay=0.15):
    engine.self_goal_adjust(metrics, k_noise=k_noise, k_decay=k_decay)
