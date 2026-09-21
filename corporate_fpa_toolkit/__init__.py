"""corporate-fpa-toolkit
=======================

A fully synthetic, driver-based FP&A (Financial Planning & Analysis) toolkit
demonstrating a budgeting -> variance -> rolling forecast -> cash flow ->
scenario/sensitivity -> capital allocation -> reporting workflow for a
multi-business-unit retail / distribution company.

.. warning::
    **All data in this package is synthetic.** Business unit names, driver
    assumptions (units, prices, costs, headcount, working-capital days,
    capex proposals, etc.) are illustrative inputs invented for this
    project. Nothing here reflects the real financial results, strategy,
    or operations of any actual company, including any company this
    package's structure is "loosely modeled on" for flavor. Do not use
    any number in this repository as a real-world reference point.

See ``README.md`` for an explanation of the FP&A concepts this package
demonstrates (driver-based budgeting, variance bridges, rolling forecasts,
indirect-method cash flow, scenario/sensitivity analysis, and capital
allocation scoring).
"""

__version__ = "0.1.0"

SYNTHETIC_DATA_DISCLAIMER = (
    "All figures produced by corporate-fpa-toolkit are synthetic model "
    "output generated from invented driver assumptions. They do not "
    "represent the real financial results of any company."
)
