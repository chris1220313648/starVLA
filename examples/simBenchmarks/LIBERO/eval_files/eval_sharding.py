"""Deterministic rollout partitioning and completeness-checked aggregation."""


def episode_indices(task_id, trials, num_shards=1, shard_id=0):
    if trials < 1 or num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError('Require positive trials/shards and 0 <= shard_id < num_shards')
    return [i for i in range(trials) if (task_id * trials + i) % num_shards == shard_id]


def summarize_records(records, num_tasks, trials, num_shards=1, shard_id=0):
    expected = {(t, i) for t in range(num_tasks)
                for i in episode_indices(t, trials, num_shards, shard_id)}
    actual = [(r['task_id'], r['episode_idx']) for r in records]
    if len(actual) != len(set(actual)):
        raise ValueError('Duplicate rollout records')
    if set(actual) != expected:
        raise ValueError(f'Incomplete or unexpected rollouts: missing={len(expected - set(actual))}, '
                         f'unexpected={len(set(actual) - expected)}')
    if any(type(r['success']) is not bool for r in records):
        raise ValueError('Each completed rollout must have a boolean success')
    tasks = []
    for task_id in range(num_tasks):
        rows = [r for r in records if r['task_id'] == task_id]
        if rows:
            successes = sum(r['success'] for r in rows)
            tasks.append(dict(task_id=task_id, task=rows[0]['task'], episodes=len(rows),
                              successes=successes, success_rate=successes / len(rows)))
    successes = sum(r['success'] for r in records)
    return dict(episodes=len(records), successes=successes,
                success_rate=successes / len(records) if records else None,
                tasks=tasks, episode_records=sorted(records, key=lambda r: (r['task_id'], r['episode_idx'])))
