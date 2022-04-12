###### Engram Cell Excitability State Determines the Efficacy of Memory Retrieval, Pignatelli et al, 2019

###### Format of Keys
# Each key corresponds to one experiment in one subfigure (e.g. 1A)
# If a subfigure involves >1 type of trial (e.g. Figure 1A has two types), each trial type has its own key (e.g. “1A-1”, “1A-2”)

###### Format of Values
# A tuple whose length is a multiple 10. The events, tests, extinctions, or recalls that occur in the trial are specified in sequential order, with each represented by 10 numerical values indicating: Time, Type, Stimulus ID, Shock, PSI, Doxycycline, Kir2.1-RFP, Rm, Rheobase, Response
# Time: Hours since time 0, with time 0 = start of first event in experiment
# Type: 0=Contextual Fear Conditioning (CFC), 1=Test, 2=Recall or Extinction, 3=Exposure, 4=Nothing
# Stimulus ID: 0=No sound/nothing, 1=Context A, 2=Context AB
# Shock: 0=absent, 1=present
# PSI: Duration of shock in seconds, -1=not applicable
# Doxycycline: 0=off, 1=on, -1=not applicable
# Kir2.1-RFP: 0=exogenous Kir_2.1 channels not expressed (RFP), 1=expressed (Kir2.1), -1=not applicable
# Rm (MOhm): Avg resting membrane potential across engram cells, -1=not applicable/not recorded
# Rheobase (pA): Avg action potential threshold across engram cells, -1=not applicable/not recorded
# Response: Percent of mice freezing to CS, -1=not applicable/not recorded

pignatelli_et_al_2016_dict = {

# Figure 1
# Figure 1: NR Group
“1-1”: (0,4,0,0,-1,0,-1,-1,-1,-1, 	24,0,1,1,-1,1,-1,-1,-1,-1,	48,4,0,0,-1,-1,-1,160,105,-1),	
# Figure 1: 5min Group
“1-2”: (0,4,0,0,-1,0,-1,-1,-1,-1, 	24,0,1,1,-1,1,-1,-1,-1,-1,	48,2,1,0,-1,-1,-1,-1,-1,-1,		48.0833,3,0,0,-1,1,-1,185,65,-1),
# Figure 1: 3hr Group
“1-2”: (0,4,0,0,-1,0,-1,-1,-1,-1, 	24,0,1,1,-1,1,-1,-1,-1,-1,	48,2,1,0,-1,-1,-1,-1,-1,-1,		51,4,0,0,-1,1,-1,150,112,-1),

# Figure 2: unrelated

# Figure 3A/B: Pattern Separation
# Figure 3A: NR Group
“3A-1”: (0,0,1,1,-1,-1,-1,-1,-1,-1,	24,1,2,0,-1,-1,-1,-1,-1,25,	48,1,2,0,-1,-1,-1,-1,-1,34,	72,1,1,0,-1,-1,-1,-1,-1,34),
# Figure 3A: 5min Group
“3A-2”: (0,0,1,1,-1,-1,-1,-1,-1,-1,	24,2,1,0,-1,-1,-1,-1,-1,36,	24.0833,1,2,0,-1,-1,-1,-1,-1,4,	48,1,2,0,-1,-1,-1,-1,-1,34,	72,1,1,0,-1,-1,-1,-1,-1,25),
# Figure 3A: 3hr Group
“3A-3”: (0,0,1,1,-1,-1,-1,-1,-1,-1,	24,2,1,0,-1,-1,-1,-1,-1,27,	27,1,2,0,-1,-1,-1,-1,-1,28,		48,1,2,0,-1,-1,-1,-1,-1,38,	72,1,1,0,-1,-1,-1,-1,-1,26),

# Figure 3C/D: Pattern Completion
# Figure 3C: NR Group, PSI 5s
“3C-1”: (0,3,1,0,-1,-1,-1,-1,-1,-1,	24,3,1,1,5,-1,-1,-1,-1,-1,	48,1,1,0,-1,-1,-1,-1,-1,11),
# Figure 3C: NR Group, PIS 10s
“3C-2”: (0,3,1,0,-1,-1,-1,-1,-1,-1,	24,3,1,1,10,-1,-1,-1,-1,-1,	48,1,1,0,-1,-1,-1,-1,-1,23),
# Figure 3C: 5min Group, PSI 5s
“3C-3”: (0,3,1,0,-1,-1,-1,-1,-1,-1,	24,2,1,0,-1,-1,-1,-1,-1,-1,	24.0833,3,1,1,5,-1,-1,-1,-1,-1,	48,1,1,0,-1,-1,-1,-1,-1,25),
# Figure 3C: 5min Group, PIS 10s
“3C-4”: (0,3,1,0,-1,-1,-1,-1,-1,-1,	24,2,1,0,-1,-1,-1,-1,-1,-1,	24.0833,3,1,1,10,-1,-1,-1,-1,-1,	48,1,1,0,-1,-1,-1,-1,-1,43),
# Figure 3C: 3hr Group, PSI 5s
“3C-5”: (0,3,1,0,-1,-1,-1,-1,-1,-1,	24,2,1,0,-1,-1,-1,-1,-1,-1,	27,3,1,1,5,-1,-1,-1,-1,-1,	48,1,1,0,-1,-1,-1,-1,-1,9),
# Figure 3C: 3hr Group, PIS 10s
“3C-6”: (0,3,1,0,-1,-1,-1,-1,-1,-1,	24,2,1,0,-1,-1,-1,-1,-1,-1,	27,3,1,1,10,-1,-1,-1,-1,-1,	48,1,1,0,-1,-1,-1,-1,-1,26),

# Figure 4D: Pattern Separation
# Figure 4D: Kir2.1-RFP- (RFP)
“4D-1”: (0,4,0,0,-1,0,0,-1,-1,-1,	24,0,1,1,-1,1,0,-1,-1,-1,	48,2,1,0,-1,1,0,-1,-1,27,	48.0833,1,2,0,-1,1,0,-1,-1,4,		72,1,2,0,-1,1,0,-1,-1,27,		96,1,1,0,-1,1,0,-1,-1,26),
# Figure 4D: Kir2.1-RFP+ (Kir2.1)
“4D-2”: (0,4,0,0,-1,0,1,-1,-1,-1,	24,0,1,1,-1,1,1,-1,-1,-1,	48,2,1,0,-1,1,1,-1,-1,31,	48.0833,1,2,0,-1,1,1,-1,-1,19,		72,1,2,0,-1,1,1,-1,-1,30,		96,1,1,0,-1,1,1,-1,-1,30),

# Figure 4E: Pattern Completion
# Figure 4E: Kir2.1-RFP- (RFP)
“4E-1”: (0,4,0,0,-1,0,0,-1,-1,-1,	24,3,1,0,-1,1,0,-1,-1,-1,		48,2,1,0,-1,1,0,-1,-1,-1,		48.0833,3,1,1,10,1,0,-1,-1,-1,		72,1,1,0,-1,1,0,-1,-1,37),
# Figure 4E: Kir2.1-RFP+ (Kir2.1)
“4E-1”: (0,4,0,0,-1,0,1,-1,-1,-1,	24,3,1,0,-1,1,1,-1,-1,-1,		48,2,1,0,-1,1,1,-1,-1,-1,		48.0833,3,1,1,10,1,1,-1,-1,-1,		72,1,1,0,-1,1,1,-1,-1,21)
}
