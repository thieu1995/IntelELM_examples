#!/usr/bin/env python
# Created by "Thieu" at 16:38, 23/10/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

PATH_SAVE = "history_new_03"

EPOCH = 1000
POP_SIZE = 50
TEST_SIZE = 0.2

LIST_METRICS = ("AS", "PS", "NPV", "RS", "F1S")

# LIST_OPTIMIZERS = ("OriginalAGTO", "OriginalAVOA", "OriginalARO", "OriginalHGSO", "OriginalEVO", "OriginalTLO")
# LIST_PARAS = [
#     {"name": "AGTO-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},
#     {"name": "AVOA-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},
#     {"name": "ARO-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},
#     {"name": "HGSO-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},
#     {"name": "EVO-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},
#     {"name": "TLO-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},
# ]
#

LIST_OPTIMIZERS = ("BaseGA", "OriginalSHADE", "L_SHADE", "RW_GWO", "HI_WOA", "OriginalSRSR",
                   "OriginalSSA", "OriginalAVOA", "OriginalAGTO", "OriginalARO", "OriginalRIME", "ImprovedQSA",
                   "OriginalSMA", "AugmentedAEO", "OrginalAOA", "OriginalINFO", "OriginalRUN")
LIST_PARAS = [
    {"name": "GA-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},
    {"name": "SHADE-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},
    {"name": "L-SHADE-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},
    {"name": "RW_GWO-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},
    {"name": "HI_WOA-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},
    {"name": "SRSR-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},

    {"name": "SSA-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},
    {"name": "AVOA-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},
    {"name": "AGTO-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},
    {"name": "ARO-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},
    {"name": "RIME-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},
    {"name": "IQSA-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},

    {"name": "SMA-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},
    {"name": "AAEO-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},
    {"name": "AOA-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},
    {"name": "INFO-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},
    {"name": "RUN-ELM", "epoch": EPOCH, "pop_size": POP_SIZE},
]
