using PicardLefschetz
using Documenter
using GLMakie

DocMeta.setdocmeta!(PicardLefschetz, :DocTestSetup, :(using PicardLefschetz); recursive=true)

makedocs(;
    modules=[
        PicardLefschetz,
        PicardLefschetz.Types,
        PicardLefschetz.Saddle,
        PicardLefschetz.Thimble,
        PicardLefschetz.DualThimble,
        PicardLefschetz.Integration
    ],
    authors="Anne Weber <anne.weber@mailbox.org>, Luvai Cutlerywala, Najma Abdullahi, and contributors",
    sitename="PicardLefschetz.jl",
    format=Documenter.HTML(;
        canonical="https://anneaux.github.io/PicardLefschetz.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => "examples.md",
        "Picard Lefschetz Theory" => "theory.md",
        "Saddle-Point Based Approach" => "saddle-point-approach.md",
        "Downward Flow" => "downward-flow.md",
        "API Reference" => "api.md"
    ],
    warnonly=[:missing_docs]
)

deploydocs(;
    repo="github.com/anneaux/PicardLefschetz.jl",
    devbranch="master",
)
