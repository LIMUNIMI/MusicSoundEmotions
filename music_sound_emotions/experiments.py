from . import settings as S
from . import validation

if __name__ == "__main__":
    obj = validation.Main()
    for p in S.RATIOS:
        S.tlog(f"Ratio: {p}")
        obj._set_p(p)
        S.tlog._log_spaces += 4
        for label in S.LABELS:
            S.tlog(f"Label: {label}")
            S.tlog._log_spaces += 4
            obj.tune_and_validate(label)
            S.tlog._log_spaces -= 4
            print("--------")
        S.tlog._log_spaces -= 4
        print("--------")
