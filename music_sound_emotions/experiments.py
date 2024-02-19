from . import settings as S
from . import validation
from .utils import telegram_notify


def full_experiment(obj, half=False):
    for p in S.RATIOS:
        obj._set_p(p)
        q = obj.splitter.q
        S.tlog(f"Ratio: {q}{obj.data1.name} + {p}{obj.data2.name}")
        S.tlog._log_spaces += 4
        for label in S.LABELS:
            S.tlog(f"Label: {label}")
            S.tlog._log_spaces += 4
            obj.tune_and_validate(label)
            S.tlog._log_spaces -= 4
            print("--------")
        S.tlog._log_spaces -= 4
        print("--------")

    if half:
        return

    telegram_notify("Swapping data!")
    obj.swap()
    for p in S.RATIOS[:-1]:  # ratio p=1 is identical...
        obj._set_p(p)
        q = obj.splitter.q
        S.tlog(f"Ratio: {q}{obj.data1.name} + {p}{obj.data2.name}")
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
    print("what is better? IADS-E no music, PMEmo, or IADS-E no music + PMEmo?")
    # Table 1 of the paper
    obj = validation.Main(("IADS-E", "PMEmo"))
    S.RATIOS = [0.0, 1.0]
    S.AUTOML_DURATION = 6 * 3600
    obj.only_automl = False
    obj.remove_iads_music = True
    obj.complementary_ratios = False
    full_experiment(obj)

    print("need to recompute for IADS with music for SOTA comparison")
    # Table 2 of the paper for IADS-E with music
    obj = validation.Main(("IADS-E", "PMEmo"))
    S.RATIOS = [0.0]
    S.AUTOML_DURATION = 6 * 3600
    obj.only_automl = True
    obj.remove_iads_music = False
    obj.complementary_ratios = False
    full_experiment(obj)

    print("HOW MUCH OF EACH DATASET SHOULD WE ADD?")
    # Table 3 of the paper
    obj = validation.Main(
        ("IADS-E-nomusic", "PMEmo"), noised="IADS-E-nomusic"
    )  # not needed, theoretically, here for avoiding possible bugs
    S.RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]
    S.AUTOML_DURATION = 3 * 3600
    obj.only_automl = True
    obj.remove_iads_music = True
    obj.complementary_ratios = False
    full_experiment(obj)

    print(
        "HOW MUCH OF EACH DATASET SHOULD WE ADD (BASELINE WITH PMEMO SHUFFLED LABELS)?"
    )
    # Table 3 of the paper
    obj = validation.Main(
        ("IADS-E-nomusic", "PMEmo"), noised="PMEmo"
    )  # not needed, theoretically, here for avoiding possible bugs
    S.RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]
    S.AUTOML_DURATION = 3 * 3600
    obj.only_automl = True
    obj.remove_iads_music = True
    obj.complementary_ratios = False
    full_experiment(obj)

    print(
        "HOW MUCH OF EACH DATASET SHOULD WE ADD (BASELINE WITH IADS-E-NOMUSIC SHUFFLED LABELS)?"
    )
    # Table 3 of the paper
    obj = validation.Main(
        ("IADS-E-nomusic-noise", "PMEmo")
    )  # not needed, theoretically, here for avoiding possible bugs
    S.RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]
    S.AUTOML_DURATION = 3 * 3600
    obj.only_automl = True
    obj.remove_iads_music = True
    obj.complementary_ratios = False
    full_experiment(obj)

    print("HOW MUCH THE ADDITION OF NEW DATA INFLUENCE THE PERFORMANCE?")
    # Table 4 of the paper (total number of data kept constant)
    obj = validation.Main(
        ("IADS-E-nomusic", "PMEmo")
    )  # not needed, theoretically, here for avoiding possible bugs
    S.RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]
    S.AUTOML_DURATION = 3 * 3600
    obj.only_automl = True
    obj.remove_iads_music = True
    obj.complementary_ratios = True
    full_experiment(obj, half=True)

    # print(
    #     "HOW MUCH THE ADDITION OF NOISY DATA INFLUENCE THE PERFORMANCE (BASELINE WITH IADS-E-NOMUSIC SHUFFLED LABELS)?"
    # )
    # # Table 5 of the paper (total number of data kept constant)
    # obj = validation.Main(
    #     ("IADS-E-nomusic", "PMEmo")
    # )  # not needed, theoretically, here for avoiding possible bugs
    # S.RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]
    # S.AUTOML_DURATION = 3 * 3600
    # S.AUTOML_DURATION = DBG_TIME
    # obj.only_automl = True
    # obj.remove_iads_music = True
    # obj.complementary_ratios = True
    # full_experiment(obj)
    #
    # print(
    #     "HOW MUCH THE ADDITION OF NOISY DATA INFLUENCE THE PERFORMANCE (BASELINE WITH IADS-E-NOMUSIC SHUFFLED LABELS)?"
    # )
    # # Table 6 of the paper (total number of data kept constant)
    # obj = validation.Main(
    #     ("IADS-E-nomusic", "PMEmo")
    # )  # not needed, theoretically, here for avoiding possible bugs
    # S.RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]
    # S.AUTOML_DURATION = 3 * 3600
    # S.AUTOML_DURATION = DBG_TIME
    # obj.only_automl = True
    # obj.remove_iads_music = True
    # obj.complementary_ratios = True
    # full_experiment(obj)
