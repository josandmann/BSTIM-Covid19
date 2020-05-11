from config import *
from plot_utils import *
from shared_utils import *
import pickle as pkl
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
from pymc3.stats import quantiles

def curves(use_interactions=True, use_report_delay=True, prediction_day=30, save_plot=False):

    with open('../data/counties/counties.pkl', "rb") as f:
        counties = pkl.load(f)

    # with open('../data/comparison.pkl', "rb") as f:
    #     best_model = pkl.load(f)

    # update to day and new limits!
    xlim = (5.5, 15.5)
    ylim = (47, 56) # <- 10 weeks

    countyByName = OrderedDict(
        [('Dortmund', '05913'), ('Leipzig', '14713'), ('Nürnberg', '09564'), ('München', '09162')])
    plot_county_names = {"covid19": ["Dortmund", "Leipzig"]}

    # colors for curves
    C1 = "#D55E00"
    C2 = "#E69F00"
    C3 = "#0073CF"

    # quantiles we want to plot
    qs = [0.25, 0.50, 0.75]

    fig = plt.figure(figsize=(12, 14))
    grid = plt.GridSpec(
        3,
        len(diseases),
        top=0.9,
        bottom=0.1,
        left=0.07,
        right=0.97,
        hspace=0.25,
        wspace=0.15,
        height_ratios=[
            1,
            1,
            1.75])

    # for i, disease in enumerate(diseases):
    i = 0
    disease = "covid19"
    prediction_region = "germany"

    data = load_daily_data(disease, prediction_region, county_info)
    first_day = data.index.min()
    last_day = data.index.max()

    _, target, _, _ = split_data(
        data,
        train_start=first_day,
        test_start=last_day - pd.Timedelta(days=1),
        post_test=last_day + pd.Timedelta(days=1))

    county_ids = target.columns

    # Load our prediction samples
    res = load_pred(disease, use_interactions, use_report_delay)
    n_days = (last_day - first_day).days - 1 #offset test data

    prediction_samples = np.reshape(res['y'], (res['y'].shape[0], n_days, -1)) 

    # TODO: figure out where quantiles comes from and if its pymc3, how to replace it
    prediction_quantiles = quantiles(prediction_samples, (5, 25, 75, 95)) 

    prediction_mean = pd.DataFrame(
        data=np.mean(
            prediction_samples,
            axis=0),
        index=target.index,
        columns=target.columns)
    prediction_q25 = pd.DataFrame(
        data=prediction_quantiles[25],
        index=target.index,
        columns=target.columns)
    prediction_q75 = pd.DataFrame(
        data=prediction_quantiles[75],
        index=target.index,
        columns=target.columns)
    prediction_q5 = pd.DataFrame(
        data=prediction_quantiles[5],
        index=target.index,
        columns=target.columns)
    prediction_q95 = pd.DataFrame(
        data=prediction_quantiles[95],
        index=target.index,
        columns=target.columns)

    # Load hhh4 predictions for reference
    # hhh4_predictions = pd.read_csv("../data/diseases/{}_hhh4.csv".format(
    #     "borreliosis_notrend" if disease == "borreliosis" else disease))
    # weeks = hhh4_predictions.pop("weeks")
    # hhh4_predictions.index = parse_yearweek(weeks)

    # create axes grid
    map_ax = fig.add_subplot(grid[2, i])
    map_ax.set_position(grid[2, i].get_position(fig).translated(0, -0.05))
    map_ax.set_xlabel(
        "{}/{}/{}".format(
            prediction_mean.index[prediction_day].year,
            prediction_mean.index[prediction_day].month,
            prediction_mean.index[prediction_day].day),
        fontsize=22)

    # plot the chloropleth map
    plot_counties(map_ax,
                counties,
                prediction_mean.iloc[prediction_day].to_dict(),
                edgecolors=dict(zip(map(countyByName.get,
                                        plot_county_names[disease]),
                                    ["red"] * len(plot_county_names[disease]))),
                xlim=xlim,
                ylim=ylim,
                contourcolor="black",
                background=False,
                xticks=False,
                yticks=False,
                grid=False,
                frame=True,
                ylabel=False,
                xlabel=False,
                lw=2)

    map_ax.set_rasterized(True)

    for j, name in enumerate(plot_county_names[disease]):
        ax = fig.add_subplot(grid[j, i])

        county_id = countyByName[name]
    #     dates = [n.wednesday() for n in target.index.values]
        dates = target.index.values

        # plot our predictions w/ quartiles
        p_pred = ax.plot_date(
            dates,
            prediction_mean[county_id],
            "-",
            color=C1,
            linewidth=2.0,
            zorder=4)
        p_quant = ax.fill_between(
            dates,
            prediction_q25[county_id],
            prediction_q75[county_id],
            facecolor=C2,
            alpha=0.5,
            zorder=1)
        ax.plot_date(
            dates,
            prediction_q25[county_id],
            ":",
            color=C2,
            linewidth=2.0,
            zorder=3)
        ax.plot_date(
            dates,
            prediction_q75[county_id],
            ":",
            color=C2,
            linewidth=2.0,
            zorder=3)


        # plot ground truth
        p_real = ax.plot_date(dates, target[county_id], "k.")

        # plot 30week marker
        ax.axvline(dates[30], lw=2)

        ax.set_title(["campylobacteriosis" if disease == "campylobacter" else disease]
                    [0] + "\n" + name if j == 0 else name, fontsize=22)
        if j == 1:
            ax.set_xlabel("Time [calendar weeks]", fontsize=22)
        ax.tick_params(axis="both", direction='out',
                    size=6, labelsize=16, length=6)
        plt.setp(ax.get_xticklabels(), visible=j > 0, rotation=45)

        cent = np.array(counties[county_id]["shape"].centroid.coords[0])
        txt = map_ax.annotate(
            name,
            cent,
            cent + 0.5,
            color="white",
            arrowprops=dict(
                facecolor='white',
                shrink=0.001,
                headwidth=3),
            fontsize=26,
            fontweight='bold',
            fontname="Arial")
        txt.set_path_effects(
            [PathEffects.withStroke(linewidth=2, foreground='black')])

        # ax.set_ylim([0,target[county_id].max()+5])
        ax.autoscale(False)
        p_quant2 = ax.fill_between(
            dates,
            prediction_q5[county_id],
            prediction_q95[county_id],
            facecolor=C2,
            alpha=0.25,
            zorder=0)
        ax.plot_date(dates, prediction_q5[county_id], ":",
                    color=C2, alpha=0.5, linewidth=2.0, zorder=1)
        ax.plot_date(dates, prediction_q95[county_id], ":",
                    color=C2, alpha=0.5, linewidth=2.0, zorder=1)

        if (i == 0) & (j == 0):
            ax.legend([p_real[0], p_pred[0], p_quant, p_quant2],
                    ["reported", "predicted", 
                        "25\%-75\% quantile", "5\%-95\% quantile"],
                    fontsize=12, loc="upper right")
        fig.text(0,
                1 + 0.025,
                r"$\textbf{" + str(i + 1) + "ABC"[j] + r"}$",
                fontsize=22,
                transform=ax.transAxes)
    fig.text(0, 0.95, r"$\textbf{" + str(i + 1) + r"C}$",
            fontsize=22, transform=map_ax.transAxes)

    fig.text(0.01, 0.66, "Reported/predicted infections",
            va='center', rotation='vertical', fontsize=22)

    if save_plot:
        plt.savefig("../figures/curves_{}_{}.pdf".format(use_interactions, use_report_delay))

    return fig


if __name__ == "__main__": 

    combinations_ia_report = [(False,False), (False,True), (True,False), (True,True)]
    for i in range(4):
        _ = curves(combinations_ia_report[i][0], combinations_ia_report[i][1],save_plot=True)

