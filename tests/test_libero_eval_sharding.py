import pytest

from examples.simBenchmarks.LIBERO.eval_files.eval_sharding import episode_indices, summarize_records


def test_eight_shards_cover_all_500_rollouts_once():
    shards = [[(task, episode) for task in range(10)
               for episode in episode_indices(task, 50, 8, shard)] for shard in range(8)]
    assert sorted(map(len, shards)) == [62] * 4 + [63] * 4
    flat = [key for shard in shards for key in shard]
    assert len(flat) == len(set(flat)) == 500
    assert set(flat) == {(task, episode) for task in range(10) for episode in range(50)}
    assert episode_indices(4, 50) == list(range(50))


@pytest.mark.parametrize('trials,shards,shard', [(0, 8, 0), (50, 0, 0), (50, 8, -1), (50, 8, 8)])
def test_bad_partition_rejected(trials, shards, shard):
    with pytest.raises(ValueError):
        episode_indices(0, trials, shards, shard)


def test_summary_counts_tasks_and_rejects_partial_or_duplicate_results():
    rows = [dict(task_id=t, episode_idx=i, task=f'task {t}', success=(i < t + 1))
            for t in range(2) for i in range(3)]
    result = summarize_records(rows, 2, 3)
    assert result['episodes'] == 6
    assert result['successes'] == 3
    assert result['success_rate'] == 0.5
    assert [r['successes'] for r in result['tasks']] == [1, 2]
    for invalid in (rows[:-1], rows + [rows[0]], rows + [dict(task_id=2, episode_idx=0, task='extra', success=True)]):
        with pytest.raises(ValueError):
            summarize_records(invalid, 2, 3)


def test_empty_shard_is_valid_but_not_a_zero_percent_result():
    result = summarize_records([], 1, 1, 8, 7)
    assert result['episodes'] == 0
    assert result['success_rate'] is None
