# Schiller et al. Nature 2010.
# Preventing the return of fear in humans using reconsolidation update mechanisms

# Format of Keys
# Each key corresponds to one experiment in one subfigure (e.g. 1A)
# If a subfigure involves >1 type of trial (e.g. Figure 1A has two types),
# each trial type has its own key (e.g. "1A-1", "1A-2")

# Format of Values
# A tuple whose length is a multiple 10. The events, tests, extinctions, or recalls that occur
# in the trial are specified in sequential order, with each represented by 10 numerical values indicating:
#   1) Time,
#   2) Type
#   3) Stimulus ID
#   4) Shock, Response Time: Hours since time 0, with time 0 = start of first event in experiment Type:
# 0=Fear Conditioning, 1=Test, 2=Retrieval, 3=Extinction, 4=Exposure, 5=LTM Test, 6=Nothing Stimulus ID: 0=No
# sound/nothing, 1=CS+, 2=CS-, 3=both CS+/-, 4=CSa+, 5=CSb- Shock: 0=absent, 1=present Response: Mean (differential)
# SCR Response, -1=not applicable/not recorded

monfils_et_al_2009_dict = {
    # Figure 1
    # Group 1: 10min
    "1-1": (0, 0, 3, 1, 0.19, 24, 2, 1, 1, -1, 24.1667, 3, 3, 0, -0.02, 48, 1, 3, 0, 0.003),
    # Group 2: 6h
    "1-2": (0, 0, 3, 1, 0.185, 24, 2, 1, 1, -1, 30, 3, 3, 0, -0.005, 48, 1, 3, 0, 0.17),
    # Group 2: no reminder
    "1-3": (0, 0, 3, 1, 0.22, 24, 3, 3, 0, -0.03, 48, 1, 3, 0, 0.135),
    # Figure 3
    "3-1": (
        0, 0, 4, 1, 0.45, 0, 0, 5, 1, 0.4, 0, 0, 2, 1, 0.17, 24, 2, 4, 1, -1, 24, 2, 2, 0, -1, 24, 6, 5, 1, -1, 24.1667,
        3,
        4, 0, 0.15, 24.1667, 3, 5, 0, 0.12, 24.1667, 3, 2, 0, 0.19, 48, 4, 0, 1, -1, 48.1667, 3, 4, 0, 0.17, 48.1667, 3,
        5,
        0, 0.51, 48.1667, 3, 2, 0, 0.19)
}
