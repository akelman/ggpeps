__all__ = [
    "system_u1_2d",
    "system_z2_2d",
    "system_z2_2d_2c",
    "system_z2_2d_G2c_F2c",
    "system_z2_2d_G2c_F4c",
    "system_z2_2d_8c",
    "system_D6_2d",
]
from ggpeps.system.system_u1_2d import U1System2DConfig, U1System2D
from ggpeps.system.system_z2_2d_1c import Z2System2DConfig
from ggpeps.system.system_z2_2d_2c import Z2System2D2CConfig
from ggpeps.system.system_z2_2d_G2c_F2c import Z2System2D_G2C_F2C_Config
from ggpeps.system.system_z2_2d_G2c_F4c import Z2System2D_G2C_F4C_Config
from ggpeps.system.system_z2_2d_8c import Z2System2D_8C_Config
from ggpeps.system.system_D6_2d import D6System2D_Config

# System2D
from ggpeps.system.system_implementation import Z2System2D

# there may be a better way to this after python 3.12
from typing import Union  # TypeAlias

SystemType = Union[
    U1System2D,
    Z2System2D,
]
SystemConfigType = Union[
    U1System2DConfig,
    Z2System2DConfig,
    Z2System2D2CConfig,
    Z2System2D_G2C_F2C_Config,
    Z2System2D_G2C_F4C_Config,
    Z2System2D_8C_Config,
    D6System2D_Config,
]
