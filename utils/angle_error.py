import numpy as np
import matplotlib.pyplot as plt


def angle():
    file_name = (
        "E://OneDrive - The University of Nottingham//Documents//Code//ToF//angle.txt"
    )
    angle = [
        "-60",
        "-50",
        "-40",
        "-30",
        "-20",
        "-10",
        "0",
        "10",
        "20",
        "30",
        "40",
        "50",
        "60",
    ]
    data = {
        "-60": [],
        "-50": [],
        "-40": [],
        "-30": [],
        "-20": [],
        "-10": [],
        "0": [],
        "10": [],
        "20": [],
        "30": [],
        "40": [],
        "50": [],
        "60": [],
    }
    with open(file_name, "r", encoding="UTF-8") as f:
        for line in f:
            if "Plane" not in line:
                angle_ = line.split(":")[-1][:-2].strip()
                for i in range(5):
                    line_ = f.readline()
                    error = float(line_.split("Error: ")[1].split("\n")[0])
                    data[angle_].append(error)
    print(data)

    up_bound = [max(data[angle]) for angle in angle]
    low_bound = [min(data[angle]) for angle in angle]
    fig = plt.figure(figsize=(14, 7))
    plt.plot([np.mean(data[angle]) for angle in angle], label="Mean Error")
    # plt.plot(up_bound, label="Upper Bound", )
    # plt.plot(low_bound, label="Lower Bound")
    plt.fill_between(range(13), up_bound, low_bound, alpha=0.3)

    ## smooth the data
    plt.xticks(range(13), angle)
    plt.xlabel("Angle")
    plt.ylabel("Error/mm")
    plt.title("Error vs Angle")

    plt.legend()
    plt.show()


def compare():
    # ours = np.array(
    #     [
    #         3.3610892698280677,
    #         3.071071350627453,
    #         3.2403698843207596,
    #         3.1959560996562555,
    #         3.498694745394526,
    #         2.9374221772147426,
    #         3.2809237010104093,
    #         4.349381721258749,
    #         3.253700170080281,
    #         3.3818851595599937,
    #     ]
    # )
    # direct = np.array(
    #     [
    #         15.160413092794583,
    #         1.7915463337884954,
    #         1.8658003282254483,
    #         14.937812316102354,
    #         14.947450486939838,
    #         14.942181922391972,
    #         1.8507121416647396,
    #         1.8067317777348466,
    #         14.939632172580927,
    #         1.9792169301967981,
    #     ]
    # )
    ours = [
        np.array(
            [
                1.0691739711942692,
                1.3018940324581902,
                1.0816878489892867,
                0.9067548921718649,
                0.618749200819651,
                0.9962697881953334,
                0.9989554177978814,
                0.7984669333360254,
                1.2790871229902825,
                0.07736914860475963,
            ]
        ),
        np.array(
            [
                0.7528694023596277,
                0.9757311871429974,
                0.9026127845844693,
                0.6555936923052406,
                0.0829736938978615,
                0.05535645054825977,
                0.3624573860888994,
                1.088010642829835,
                0.8131537610362104,
                0.44019322710542497,
            ]
        ),
        np.array(
            [
                0.34307321304175703,
                0.48263760409013345,
                0.02057319546467451,
                0.485308086357144,
                0.3507863238549589,
                0.1236968338522037,
                0.30524873473896946,
                0.1403763300392568,
                0.1511364959355809,
                0.3620382159620466,
            ]
        ),
    ]
    direct = [
        np.array(
            [
                5.910562957432781,
                8.017175889871279,
                6.882134333723208,
                7.755802705670139,
                7.609545969015615,
                7.518510271258437,
                7.416334160713976,
                7.32182617097136,
                7.2715207084843225,
                7.284344070715293,
            ]
        ),
        np.array(
            [
                6.617442695295378,
                13.139190967435162,
                12.883969915832571,
                11.701735958012735,
                12.190546118003946,
                11.895503679158638,
                11.725176495122504,
                10.561893280051729,
                10.730307267130039,
                11.703366962241583,
            ]
        ),
        np.array(
            [
                8.713990896140018,
                2.469273762111803,
                2.445249815566542,
                2.2861680049088866,
                2.2749371152213986,
                2.3204030802264652,
                2.39604317191181,
                2.3767909514957197,
                2.3052374450536144,
                2.6038763204520614,
            ]
        ),
    ]
    # plt.scatter(x=[1 for _ in range(10)], y=ours, label="Ours")
    # plt.scatter(x=[2 for _ in range(10)], y=direct, label="Direct")
    # plt.legend()

    plt.bar(
        x=[1, 3, 5],
        height=[np.mean(ours[i]) for i in range(3)],
        yerr=[np.std(ours[i]) for i in range(3)],
        label="Ours",
        width=0.5,
    )
    plt.bar(
        x=[1.5, 3.5, 5.5],
        height=[np.mean(direct[i]) for i in range(3)],
        yerr=[np.std(direct[i]) for i in range(3)],
        label="Direct",
        width=0.5,
    )
    plt.legend()
    plt.xticks([1.25, 3.25, 5.25], ["1", "2", "3"])
    plt.ylabel("RMSE Error/mm")
    plt.title("Error Comparison")
    plt.show()
    print(np.mean(ours, axis=1))
    print(np.mean(direct, axis=1))


if __name__ == "__main__":
    compare()
