using DataFrames, CSV, XLSX, OrderedCollections, CairoMakie, PairPlots

# Load Table with column descriptions and units
dataref = let reference = OrderedDict()
    path = abspath(joinpath(pwd(),"data/pdp_units_data.xlsx"))
    xlsxfile = XLSX.readxlsx(path)
    for title in XLSX.sheetnames(xlsxfile)
        reference[title] = DataFrame(XLSX.gettable(xlsxfile[title]))
    end
    reference
end

# Load PDP Data Set
dataset = let 
	path = abspath(joinpath(pwd(),"data/Perovskite_database_content_all_data_040524.csv"))
	DataFrame(CSV.File(path))
end


properties = [
    # Stabilised PCE Performance
    "Stabilised_performance_PCE",
    # Reverse scan JV
    "JV_reverse_scan_Voc",
    "JV_reverse_scan_Jsc",
    "JV_reverse_scan_FF",
    "JV_reverse_scan_PCE",
    # Forward scan JV
    "JV_forward_scan_Voc",
    "JV_forward_scan_Jsc",
    "JV_forward_scan_FF",
    "JV_forward_scan_PCE"
]

reducedset = dataset[!, properties]

# Filter missing data from the reduced data set
filteredset = dropmissing(reducedset) # Remove all missing data
# filteredset = dropmissing(reducedset,["Stabilised_performance_PCE", "JV_reverse_scan_PCE", "JV_forward_scan_PCE"])

stabilised_pce = filteredset[!, 1]

reverse_voc = filteredset[!, 2]
reverse_jsc = filteredset[!, 3]
reverse_ff = filteredset[!, 4]
reverse_pce = filteredset[!, 5]

forward_voc = filteredset[!, 6]
forward_jsc = filteredset[!, 7]
forward_ff = filteredset[!, 8]
forward_pce = filteredset[!, 9]

function plotregression!(f, args...;
    color=:blue, xlabel="", ylabel="", title="", clabel="", ticks=Makie.automatic,
    qwargs...
    )
    ax, sc = scatter(f[1, 1], args...; color=color, qwargs...)
    ax.aspect = 1
    ax.xlabel = xlabel
    ax.ylabel = ylabel
    ax.title = title
    ax.xticks = 0:5:25
    ax.yticks = 0:5:25
    lines!(ax, [0,25.5], [0,25.5], color=:red, linewidth=2)
    if typeof(color) <: Vector
        Colorbar(f[1,2], sc; size=30, label=clabel, ticks=ticks)
    end
    f
end
function plotregression(args...; size=(600,500), qwargs...)
    f = Figure(size=size)
    plotregression!(f, args...; qwargs...)
end

finalfig = let fig = Figure(size=(900,750))
    ga = fig[1,1] = GridLayout()
    gb = fig[1,2] = GridLayout()
    gc = fig[2,1] = GridLayout()
    gd = fig[2,2] = GridLayout()

    plotregression!(ga, reverse_pce, stabilised_pce;
        xlabel="Reverse Scan PCE", ylabel="Stabilised PCE",
        ticks=0:0.2:1.5,
    )

    plotregression!(gb, reverse_pce, stabilised_pce;
        color=reverse_voc,
        xlabel="Reverse Scan PCE", ylabel="Stabilised PCE",
        clabel="Voc",
        ticks=0:0.2:1.5,
        colormap=:thermal
    )

    plotregression!(gc, reverse_pce, stabilised_pce;
        color=reverse_pce-forward_pce,
        xlabel="Reverse Scan PCE", ylabel="Stabilised PCE",
        clabel="Reverse and Forward PCE Difference",
        ticks=-5:5:20,
        colormap=:vikO10
    )

    plotregression!(gd, reverse_pce, forward_pce;
        color=reverse_ff-forward_ff,
        xlabel="Reverse Scan PCE", ylabel="Forward PCE",
        clabel="Reverse and Forward FF Difference",
        ticks=-0.2:0.2:0.7,
        colormap=:vikO10
    )
    for (label, layout) in zip(["A", "B", "C", "D"], [ga, gb, gc, gd])
        Label(layout[1, 1, TopLeft()], label,
            fontsize = 26,
            font = :bold,
            padding = (0, 5, 5, 0),
            halign = :right)
    end
    fig
end

cornerfig = pairplot(filteredset)

namespce = properties[[1,5,9]]
labelpce = Dict(Symbol.(namespce) .=> ["Stabilised PCE", "Reverse Scan PCE", "Forward Scan PCE"])
pceset = filteredset[!, namespce]
pairplot(pceset => (
    PairPlots.MarginDensity(),
    PairPlots.MarginHist(),
    PairPlots.MarginConfidenceLimits(quantiles=(0.1, 0.3, 0.9)),
    PairPlots.Contour(),
    PairPlots.Scatter(filtersigma=2)
); labels=labelpce)

cornerpce = let fig = Figure(size=(1000,900))
    gs = GridLayout(fig[1,1])

    pairplot(gs, pceset => (
        PairPlots.Scatter(color=reverse_ff-forward_ff, markersize=10, colormap=:vikO10),
        PairPlots.TrendLine(color=:red),
        PairPlots.Correlation(),
        PairPlots.MarginDensity(),
        PairPlots.MarginHist(),
        PairPlots.MarginConfidenceLimits(),
    ); labels=labelpce)
    rowgap!(gs, 0)
    colgap!(gs, 0)
    Colorbar(fig[1,2];
        limits=extrema(reverse_ff-forward_ff),
        colormap=:vikO10,
        ticks=-0.2:0.2:0.7,
        size=30,
        label="Reverse and Forward FF Difference",
        flipaxis= true
    )
    fig
end