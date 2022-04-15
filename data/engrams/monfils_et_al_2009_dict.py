# Monfils et al. Science 2009
# Extinction-Reconsolidation Boundaries: Key to Persistent Attenuation of Fear Memories,

# Format of Keys Each key corresponds to one experiment in one subfigure (e.g. 1A) If a subfigure involves >1
# type of trial (e.g. Figure 1A has two types), each trial type has its own key (e.g. "1A-1", "1A-2")

# Format of Values
# A tuple whose length is a multiple 6. The events, tests, extinctions, or recalls
# that occur in the trial are specified in sequential order, with each represented
# by 6 numerical values indicating:
#   1) Time: Hours since time 0
#   2) Type: 0=Fear Conditioning, 1=Test, 2=Retrieval, 3=Extinction, 4=Exposure, 5=LTM Test, 6=Nothing
#   3) Stimulus ID: 0=No sound/nothing, 1=Context A, 2=Context B
#   4) Number of CS: 1, 2, 3, 4, or 5, -1=not applicable
#   5) Shock: 0=absent, 1=present
#   6) Response: Percent of mice freezing to CS, -1=not applicable/not recorded

# Assumption: 1 month in paper = 30 days

monfils_et_al_2009_dict = {
    # Figure1
    # No Ret
    "1-1": ((0, 0, 1, 1, 1, -1),
            (24, 3, 1, 1, 0, 19),
            (48, 5, 1, 1, 0, 25),
            (768, 1, 1, 1, 0, 38)),
    # Ret, 10min
    "1-2": (
    0, 0, 1, 1, 1, -1, 24, 2, 1, 1, 0, -1, 24.1667, 3, 1, 1, 0, 21, 48.1667, 5, 1, 1, 0, 26, 768.1667, 1, 1, 1, 0, 26),
    # Ret, 1hr
    "1-3": (0, 0, 1, 1, 1, -1, 24, 2, 1, 1, 0, -1, 25, 3, 1, 1, 0, 17, 49, 5, 1, 1, 0, 23, 769, 1, 1, 1, 0, 16),
    # Ret, 6hr
    "1-4": (0, 0, 1, 1, 1, -1, 24, 2, 1, 1, 0, -1, 30, 3, 1, 1, 0, 20, 54, 5, 1, 1, 0, 28, 774, 1, 1, 1, 0, 55),
    # Ret, 24hr
    "1-5": (0, 0, 1, 1, 1, -1, 24, 2, 1, 1, 0, -1, 48, 3, 1, 1, 0, 24, 72, 5, 1, 1, 0, 22, 792, 1, 1, 1, 0, 38),

    # Figure2
    # No Ret
    "2-1": (0, 0, 1, 1, 1, -1, 24, 6, 2, 1, 0, -1, 25, 3, 2, 1, 0, 10, 49, 5, 2, 1, 0, 10, 769, 1, 1, 1, 0, 25),
    # Ret, 1hr
    "2-2": (0, 0, 1, 1, 1, -1, 24, 2, 2, 1, 0, 54, 25, 3, 2, 1, 0, 10, 49, 5, 2, 1, 0, 9, 769, 1, 1, 1, 0, 5),

    # Figure3
    # No Ret
    "3-1": (0, 0, 1, 1, 1, -1, 24, 6, 1, 1, 0, -1, 25, 3, 1, 1, 0, 13, 49, 7, 1, 1, 1, -1, 73, 1, 1, 1, 0, 46),
    # Ret, 1hr
    "3-2": (0, 0, 1, 1, 1, -1, 24, 2, 1, 1, 0, 54, 25, 3, 1, 1, 0, 21, 49, 7, 1, 1, 1, -1, 73, 1, 1, 1, 0, 20),

    # Figure5
    # No Ret
    "5-1": (0, 0, 1, 3, 1, -1, 24, 6, 1, 1, 0, -1, 25, 3, 1, 1, 0, 5, 49, 0, 1, 1, 1, 8, 73, 5, 1, 1, 0, 34),
    # Ret, 1hr
    "5-2": (0, 0, 1, 3, 1, -1, 24, 2, 1, 1, 0, 56, 25, 3, 1, 1, 0, 10, 49, 0, 1, 1, 1, 11, 73, 5, 1, 1, 0, 20),

    # Figure6
    # No Ret
    "6-1": (
    0, 0, 1, 5, 1, -1, 24, 6, 1, 1, 0, -1, 25, 3, 1, 1, 0, -1, 49, 0, 1, 1, 1, 15, 49, 0, 1, 1, 1, 53, 49, 0, 1, 1, 1,
    50, 49, 0, 1, 1, 1, 65, 49, 0, 1, 1, 1, 50),
    # Ret, 1hr
    "6-2": (
    0, 0, 1, 5, 1, -1, 24, 2, 1, 1, 0, 56, 25, 3, 1, 1, 0, -1, 49, 0, 1, 1, 1, 10, 49, 0, 1, 1, 1, 26, 49, 0, 1, 1, 1,
    37, 49, 0, 1, 1, 1, 34, 49, 0, 1, 1, 1, 35)
}
