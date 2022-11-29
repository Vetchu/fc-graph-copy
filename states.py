from abc import ABC

import pandas as pd
from FeatureCloud.app.engine.app import app_state, Role
from FeatureCloud.app.engine.library import ConfigState

from prepare_file_for_fc import read_csv

# from utils.pytorch.states import Initialization, LocalUpdate, GlobalAggregation, WriteResults

name = 'fc-cfr-test'


class Initialization(ConfigState.State, ABC):
    """
    Read input data
    read config files

    """

    def run(self) -> str or None:
        self.update(state=op_state.RUNNING)
        self.update(message="Reading the config file....")
        self.read_config()
        self.lazy_init()
        self.finalize_config()
        self.store('iteration', 0)


@app_state(name='initial', role=Role.BOTH, app_name=name)
class B1(Initialization):
    """
    Read input data
    read config files

    """

    def register(self):
        self.register_transition('Local_Update')

    def run(self) -> str or None:
        super().run()
        return 'Local_Update'


from FeatureCloud.app.engine.app import State as op_state


@app_state('Local_Update', Role.BOTH)
class B2:
    """ Local Model training
        Input:
            Model weights(Coordinator already has it)
            App statuses: {Converged: True/False }
    """

    # LocalUpdate
    def register(self):
        self.register_transition('Global_Aggregation', Role.COORDINATOR)
        self.register_transition('Local_Update', Role.PARTICIPANT)
        self.register_transition('Write_Results', Role.PARTICIPANT)

    def run(self) -> str or None:
        msg = read_csv()
        # msg = super().run()
        if msg is not None:
            return 'Write_Results'
        if self.is_coordinator:
            return 'Global_Aggregation'
        return 'Local_Update'

# @app_state('Global_Aggregation', Role.COORDINATOR)
# class C1(GlobalAggregation):
#     def register(self):
#         self.register_transition('Local_Update', Role.COORDINATOR)
#         self.register_transition('Write_Results', Role.COORDINATOR)
#
#     def run(self) -> str or None:
#         smg = super().run()
#         if smg is not None:
#             return 'Write_Results'
#         return 'Local_Update'
#
#
# @app_state('Write_Results', Role.BOTH)
# class B3(WriteResults):
#     def register(self):
#         self.register_transition('terminal')
#
#     def run(self) -> str or None:
#         super().run()
#         return 'terminal'
