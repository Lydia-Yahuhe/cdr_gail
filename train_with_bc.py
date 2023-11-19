from data.data_set import OurSet

from env.environment import ConflictEnv
from parameters import *

from baselines.deepq.behavior_clone import BehaviorClone


def train():
    print('Run train with bc')
    env = ConflictEnv(limit_path=limit_path, size=size, ratio=ratio)
    expert_set = OurSet(expert_path=expert_path)
    learner = BehaviorClone(env, expert_set)
    learner.learn(
        batch_size=common_params['batch_size'],
        max_iters=common_params['total_timesteps'],
        lr=common_params['lr'],
        verbose=True,
        save_path=bc_actions_path
    )
    env.close()


def test(start=0, end=100001, delta=5000):
    print('Run test with bc')
    env = ConflictEnv(limit_path=limit_path, size=size, ratio=ratio)
    expert_set = OurSet(expert_path=expert_path)
    learner = BehaviorClone(env, expert_set)

    for step in range(start, end, delta):
        print('{}/{}'.format(step, end - 1))
        load_path = bc_actions_path + 'my_model_{}.pkl'.format(step)
        print('Loaded model from {}'.format(load_path))

        act = learner.load(path=load_path)
        ob_expert, ac_expert = learner.expert_data_set.get_next_batch(learner.test_set)
        val_loss, _ = learner.lossandgrad(ob_expert, ac_expert)
        print(val_loss)

        env.evaluate(
            act,
            save_path=bc_actions_path + 'bc_actions_train_{}'.format(step),
            use_set='train'
        )
        env.evaluate(
            act,
            save_path=bc_actions_path + 'bc_actions_test_{}'.format(step),
            use_set='test'
        )
    env.close()


if __name__ == '__main__':
    train()
    # test()
