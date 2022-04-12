###### Format of Keys
# Each key corresponds to one experiment in one subfigure (e.g. 1A)
# If a subfigure involves >1 type of trial (e.g. Figure 1A has two types), 
#    then each trial type has its own key (e.g. “1A-1”, “1A-2”)

###### Format of Values
# A tuple whose length is a multiple 7. The events, tests, extinctions, or recalls that 
#    occur in the trial are specified in sequential order, with each represented by 7 
#    numerical values indicating: 
#    Time, Type, Stimulus ID, Shock, Excitation, Inhibition, Response
#
# Time: a number representing hours since time 0, time 0 = start of first event in experiment
# Type: 0=event, 1=test, 2=recall or extinction
# Stimulus ID: 0 for no sound, 1 for CS1, 2 for CS2, 3 for CS3
# Shock: 0 if absent, 1 if present
# Neuron excitation (blue light): 0 if not excited, 1 if excited 
# Neuron inhibition (red light): 0 if not inhibited, 1 if inhibited 
# Response: percent of mice freezing to the CS, or -1 if not applicable/not recorded


rashid_et_al_2016_dict = {
# Figure 1A
“1A-1”: (0,0,1,1,0,0,-1, 	24,1,1,0,0,0,0.58,	48,1,2,0,0,0,0.21),
“1A-2”: (0,0,2,1,0,0,-1, 	24,1,1,0,0,0,0.2, 	48,1,2,0,0,0,0.50),
# Figure 1B: CS1 alone
“1B-1”: (0,0,1,0,0,0,-1, 	1.5,0,2,1,0,0,-1, 	25.5,1,2,0,0,0,0.35),
“1B-2”: (0,0,1,0,0,0,-1, 	3,0,2,1,0,0,-1, 	27,1,2,0,0,0,0.40),
“1B-3”: (0,0,1,0,0,0,-1, 	6,0,2,1,0,0,-1, 	30,1,2,0,0,0,0.40),
“1B-4”: (0,0,1,0,0,0,-1, 	18,0,2,1,0,0,-1, 	42,1,2,0,0,0,0.40),
“1B-5”: (0,0,1,0,0,0,-1, 	24,0,2,1,0,0,-1, 	48,1,2,0,0,0,0.40),
# Figure 1B: CS1+shock
“1B-6”: (0,0,1,1,0,0,-1, 	1.5,0,2,1,0,0,-1, 	25.5,1,2,0,0,0,0.68),
“1B-7”: (0,0,1,1,0,0,-1, 	3,0,2,1,0,0,-1, 	27,1,2,0,0,0,0.65),
“1B-8”: (0,0,1,1,0,0,-1, 	6,0,2,1,0,0,-1,  	30,1,2,0,0,0,0.69),
“1B-9”: (0,0,1,1,0,0,-1, 	18,0,2,1,0,0,-1,  	42,1,2,0,0,0,0.56),
“1B-10”:(0,0,1,1,0,0,-1,	24,0,2,1,0,0,-1, 	48,1,2,0,0,0,0.45),
# Figure 1B: Imm. shock
“1B-11”: (0,0,0,1,0,0,-1, 	6,0,2,1,0,0,-1,		30,1,2,0,0,0,0.46),
“1B-12”: (0,0,0,1,0,0,-1,   24,-,2,1,0,0,-1,	48,1,2,0,0,0,0.43),

# Figure 1C: CS1-CS1
# “1C-1”: (0,0,1,1,0,0,-1, 	24,1,1,0,0,0,-1, 	24.41666667,1,2,0,0,0,0.58),
# Figure 1C: CS1-CS3
# “1C-2”: (0,0,1,1,0,0,-1, 	24,1,1,0,0,0,-1, 	24.41666667,1,3,0,0,0,0.07),
# Figure 1C: CS1-CS2
“1C-3”: (0,0,1,1,0,0,-1, 	6,0,1,1,0,0,-1, 	30,1,1,0,0,0,-1, 	30.41666667,1,3,0,0,0,0.30),
“1C-4”: (0,0,1,1,0,0,-1,	24,0,1,1,0,0,-1 	48,1,1,0,0,0,-1, 	48.41666667,1,3,0,0,0,0.13),

# Figure 1D: CS1-CS2, ITI 6h
“1D-1”: (0,0,1,1,0,0,-1, 	6,0,2,1,0,0,-1, 	30,1,1,0,0,0,0.54, 	54,1,2,0,0,0,0.68,	54,2,2,0,0,0,-1,	78,1,2,0,0,0,0.37,	102,1,1,0,0,0,0.40),
# Figure 1D: CS1-CS2, ITI 24h
“1D-2”: (0,0,1,1,0,0,-1, 	24,0,2,1,0,0,-1, 	48,1,1,0,0,0,0.51, 	72,1,2,0,0,0,0.50,	96,2,2,0,0,0,-1,	120,1,2,0,0,0,0.40,		144,1,1,0,0,0,0.48),
# Figure 1D: CS1+shock, ITI 6h
“1D-3”: (0,0,1,1,0,0,-1, 	6,0,2,0,0,0,-1, 	30,1,1,0,0,0,0.55, 	54,1,2,0,0,0,0.10,	54,2,2,0,0,0,-1,	78,1,2,0,0,0,0.15,	102,1,1,0,0,0,0.53),

# Figure 2: N/A

# Figure 3
# Figure 3A: N/A
# Figure 3B
# NpACY BL-, RL+
“3B-1”: (0,0,1,1,0,0,-1, 	24,1,1,0,0,1,0.53),
# NpACY BL+, RL+
“3B-2”: (0,0,1,1,1,0,-1, 	24,1,1,0,0,1,0.27),
# NpACY BL-, RL-
“3B-3”: (0,0,1,1,0,0,-1, 	24,1,1,0,0,0,0.50),
# NpACY BL+, RL-
“3B-4”: (0,0,1,1,1,0,-1, 	24,1,1,0,0,0,0.58),

# Figure 3C
# ITI 6h, RL+
“3C-1”: (0,0,1,1,1,0,-1, 	6,0,2,1,0,0,-1, 	30,1,1,0,0,1,0.29,	54,1,2,0,0,1,0.27),
# ITI 6h, RL-
“3C-2”: (0,0,1,1,1,0,-1, 	6,0,2,1,0,0,-1, 	30,1,1,0,0,0,0.48,	54,1,2,0,0,0,0.53),
# ITI 24h, RL+
“3C-3”: (0,0,1,1,1,0,-1, 	24,0,2,1,0,0,-1, 	48,1,1,0,0,1,0.29,	72,1,2,0,0,1,0.44),
# ITI 24h, RL-
“3C-4”: (0,0,1,1,1,0,-1, 	24,0,2,1,0,0,-1, 	48,1,1,0,0,0,0.48,	72,1,2,0,0,0,0.40),

# Figure 3D
# RL+
“3D-1”: (0,0,1,1,1,0,-1, 	24,0,2,1,1,0,-1,	48,1,1,0,0,1,0.29,	72,1,2,0,0,1,0.34),
# RL-	
“3D-2”: (0,0,1,1,1,0,-1, 	24,0,2,1,1,0,-1,	48,1,1,0,0,0,0.55,	72,1,2,0,0,0,0.53),

# Figure 3E
# ITI 6h, RL+
“3E-1”: (0,0,1,1,1,0,-1, 	6,0,2,1,0,1,-1,		30,1,1,0,0,1,0.30,	54,1,2,0,0,1,0.25),
# ITI 6h, RL-
“3E-2”: (0,0,1,1,1,0,-1, 	6,0,2,1,0,1,-1,		30,1,1,0,0,0,0.57,	54,1,2,0,0,0,0.19),
# ITI 24h, RL+
“3E-3”: (0,0,1,1,1,0,-1, 	24,0,2,1,0,1,-1,	48,1,1,0,0,1,0.30,	72,1,2,0,0,1,0.47),
# ITI 24h, RL-
“3E-4”: (0,0,1,1,1,0,-1, 	24,0,2,1,0,1,-1,	48,1,1,0,0,0,0.57,	72,1,2,0,0,0,0.44),

# Figure 4
# Figure 4A: N/A
# Figure 4B: N/A; same progression as 3E aside from suppression of interneurons before Event 2

# Figure 4C
# RL+
“4C-1”: (0,0,1,1,0,1,-1,	6,0,2,1,1,0,-1,		30,1,1,0,0,1,0.50,	54,1,2,0,0,1,0.31),
# RL-
“4C-2”: (0,0,1,1,0,1,-1,	6,0,2,1,1,0,-1,		30,1,1,0,0,0,0.49,	54,1,2,0,0,0,0.53),

# Figure 4D
# Recall CS1, ITI 6h
“4D-1”: (0,0,1,1,0,0,-1,	24,2,1,0,0,0,-1,	30,0,2,1,0,0,-1,	54,1,2,0,0,0,0.64),
# Recall CS1, ITI 24h
“4D-2”: (0,0,1,1,0,0,-1,	24,2,1,0,0,0,-1,	48,0,2,1,0,0,-1,	72,1,2,0,0,0,0.47),
# No Recall CS1, ITI 6h
“4D-3”: (0,0,1,1,0,0,-1,	24,0,0,0,0,0,-1,	30,0,2,1,0,0,-1,	54,1,2,0,0,0,0.43),
}

