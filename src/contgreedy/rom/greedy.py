"""Outer greedy orchestration; numerical operations belong to the supplied callbacks."""

from time import perf_counter


def greedy(
    state,
    *,
    rounds,
    select,
    truth,
    enrich,
    rank,
    observe=None,
    stop=None,
    stop_on_dependent=False,
):
    """Return state and records, keeping rounds, rank and truth calls distinct.

    select must synchronize its returned point before returning. stop is checked
    before each selection; stop_on_dependent is an explicit per-example policy.
    RNG streams and pool lifetimes are owned by the caller.
    """
    if rounds < 0:
        raise ValueError("rounds must be nonnegative")
    history = []
    for iteration in range(rounds):
        if stop is not None and stop(state):
            break
        before = rank(state)
        start = perf_counter()
        parameter, info = select(state, iteration)
        search_seconds = perf_counter() - start
        start = perf_counter()
        snapshot = truth(parameter)
        truth_seconds = perf_counter() - start
        start = perf_counter()
        state = enrich(state, snapshot)
        row = dict(
            info,
            round=iteration + 1,
            rank_before=before,
            rank_after=rank(state),
            truth_calls=1,
            search_seconds=search_seconds,
            truth_seconds=truth_seconds,
            enrichment_seconds=perf_counter() - start,
        )
        if observe is not None:
            row["audit"] = observe(state, parameter)
        history.append(row)
        if stop_on_dependent and row["rank_after"] == before:
            break
    return state, history
