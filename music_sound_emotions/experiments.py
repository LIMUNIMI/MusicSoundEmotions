from . import settings as S
from . import validation
from .utils import telegram_notify

if __name__ == "__main__":
    obj = validation.Main(("iads", "pmemo"))
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

    telegram_notify("Swapping data!")
    obj.splitter.swap()
    for p in S.RATIOS[:-1]:  # ratio p=1 is identical...
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
    telegram_notify("Ended!")
