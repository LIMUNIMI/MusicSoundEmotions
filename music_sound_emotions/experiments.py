from . import settings as S
from . import validation
from .utils import telegram_notify


def full_experiment(obj):
    for p in S.RATIOS:
        S.tlog(f"Ratio: {obj.data1.name} + {p}{obj.data2.name}")
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
    obj.swap()
    for p in S.RATIOS[:-1]:  # ratio p=1 is identical...
        S.tlog(f"Ratio: {obj.data1.name} + {p}{obj.data2.name}")
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


if __name__ == "__main__":
    # what is better? IADS-E no music, PMEmo, or IADS-E no music + PMEmo?
    # Table 1 of the paper
    obj = validation.Main(("IADS-E", "PMEmo"))
    S.RATIOS = [0.0, 1.0]
    S.AUTOML_DURATION = 8 * 3600
    obj.only_automl = False
    obj.remove_iads_music = True
    obj.complementary_ratios = False
    full_experiment(obj)

    # need to recompute for IADS with music for SOTA comparison
    # Table 2 of the paper for IADS-E with music
    obj = validation.Main(("IADS-E", "PMEmo"))
    S.RATIOS = [0.0]
    S.AUTOML_DURATION = 8 * 3600
    obj.only_automl = True
    obj.remove_iads_music = False
    obj.complementary_ratios = False
    full_experiment(obj)

    # how much of each dataset should we add?
    # Table 3 of the paper
    obj = validation.Main(
        ("IADS-E-nomusic", "PMEmo")
    )  # not needed, theoretically, here for avoiding possible bugs
    S.RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]
    S.AUTOML_DURATION = 4 * 3600
    obj.only_automl = True
    obj.remove_iads_music = True
    obj.complementary_ratios = False
    full_experiment(obj)

    # how much the addition of new data influence the performance?
    # Table 4 of the paper (total number of data kept constant)
    obj = validation.Main(
        ("IADS-E-nomusic", "PMEmo")
    )  # not needed, theoretically, here for avoiding possible bugs
    S.RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]
    S.AUTOML_DURATION = 4 * 3600
    obj.only_automl = True
    obj.remove_iads_music = True
    obj.complementary_ratios = True
    full_experiment(obj)
