from . import settings as S
from . import validation

obj = validation.Main()
for p in S.RATIOS:
    obj._set_p(p)
    for label in S.LABELS:
        obj.tune_and_validate(label)
