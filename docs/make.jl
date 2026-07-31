using PicardLefschetz
using Documenter

DocMeta.setdocmeta!(PicardLefschetz, :DocTestSetup, :(using PicardLefschetz); recursive=true)

makedocs(;
    modules=[PicardLefschetz],
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
        "Theory" => "theory.md",
        "Saddle-Point Based Approach" => "saddle-point-approach.md"
    ],
    warnonly=[:missing_docs]
)

deploydocs(;
    repo="github.com/anneaux/PicardLefschetz.jl",
    devbranch="master",
)
