#=
    This file parses the Perovskite Database Description to create a reference table for the Data Fields
    [Field, Format, Default, Unit, Implemented, Description, Concerns]
=#
using DataFrames, OrderedCollections, CSV, AbstractTrees, XLSX, JSON

struct computeTree
    key::String
    text::String
    children::Vector{computeTree}
end
AbstractTrees.children(t::computeTree) = t.children
AbstractTrees.nodevalue(t::computeTree) = t.text

# Each Section has a prefix for the Data Fields
const SECTIONS_KEYS = OrderedDict(
    "Reference information" => "Ref",
    "Cell definition" => "Cell",
    "Module definition" => "Module",
    "Substrate" => "Substrate",
    "Electron transport layer" => "ETL",
    "The perovskite" => "Perovskite",
    "Perovskite deposition" => "Perovskite",
    "Hole transport layer" => "HTL",
    "Back contact" => "Backcontact",
    "Additional layers" => "Add_lay",
    "Encapsulation" => "Encapsulation",
    "JV data" => "JV",
    "Stabilised efficiency" => "Stabilised",
    "Quantum efficiency" => "EQE",
    "Stability" => "Stability",
    "Outdoor testing" => "Outdoor",
)

const TABLE_KEYS = Dict(
    "Format: " => "Type",
    "Default: " => "Default",
    "Implemented: " => "Implemented",
    "Description: " => "Description",
    "Concerns: " => "Concerns",
)
const TABLE_KEY_ARR = collect(pairs(TABLE_KEYS))

const TYPE_KEYS = Dict(
    "integer" => "Integer",
    "string" => "String",
    "boolean" => "Bool",
    "float" => "Float",
    "date" => "Date"
)

const DEFAULT_KEYS = Dict(
    "string" => "\"\"",
    "unknown" => "Unknown",
    "false" => false
)

# Extract lines into sections
filepath = joinpath(pwd(), "data\\raw\\pdp_description_5.4.txt")
description, filestring = open(filepath, "r") do file
    sections = collect(keys(SECTIONS_KEYS))
    description = OrderedDict()
    section = ""
    field = ""
    filestring = ""
    debug = 0
    println("_________Opening File_________")
    for line in eachline(file)
        line = replace(line, "‘" => "'")
        line = replace(line, "’" => "'")
        filestring *= line
        words = split(line, " ") # Sometimes the Fields are not on their own line or the section key occurs in a description
        
        if in(line, sections)
            # Filter by each section
            filter!(x->x!=line, sections)
            section = line
            description[section] = OrderedDict()
        elseif occursin(SECTIONS_KEYS[section], first(words))
            # Filter by each Data Field
            # Runs if the first word of a line occurs in the section keys
            field = first(words)
            description[section][field] = []
            if length(words) > 1
                # remove the field and push the line data to it's dictionary
                info = replace(line, "$(field) " => "")
                push!(description[section][field], info)
            end
        else
            push!(description[section][field], line)
        end
    end
    description, filestring
end

# find all units and patterns
function unitfilter(x)
    # Hand tuned to extract all the units
    if length(x) > 30
        return false
    elseif any(occursin.([";", "|", "…", ".constant", "Substrate", "0/1/2", "between"], x))
        return false
    else
        return true
    end
end
function findunitpattern(x)
    unitpattern = unique([match.match for match in eachmatch(r"\[(.*?)]", x)])
    units = filter(unitfilter, unitpattern)
    patterns = setdiff(unitpattern, units)
    return units, patterns
end

# define functions for seperating column information [Format, Default, Implemented, Description, Concerns]
function splitcolumns(text::String; i::Int=1, depth::Int=length(TABLE_KEY_ARR), tree=computeTree("", text, []))
    key = first(TABLE_KEY_ARR[i])
    substrings = split(text, key)
    for (j, sub) in enumerate(substrings)
        if j == 1
            child = computeTree(tree.key, sub, [])
        else
            child = computeTree(key, sub, [])
        end
        push!(children(tree), child)
        if i < depth
            splitcolumns(string(sub); i=i+1, tree=child)
        end
    end
    return collect(Leaves(tree))
end

function separatecolumns(lines::Vector)
    text = mapreduce(x->x*" ", *, lines)
    leaves = splitcolumns(text)
    table = Dict()

    for leaf in leaves
        if haskey(TABLE_KEYS, leaf.key)
            key = TABLE_KEYS[leaf.key]
            text = leaf.text
            if key == "Type"
                units, patterns = findunitpattern(text)
                unit = !isempty(units) ? first(units) : ""
                pattern = !isempty(patterns) ? first(patterns) : ""
                text = replace(text, unit=>"")
                text = replace(text, pattern=>"")
                text = replacekeys(text, TYPE_KEYS)
                
                table["Unit"] = unit
                table["Pattern"] = pattern
            elseif key == "Default"
                text = replacekeys(text, DEFAULT_KEYS)
            end
            table[key] = text
        end
    end

    # If a column is not present in the data it is added as an empty string
    tablekeys = collect(keys(table))
    refvalues = collect(values(TABLE_KEYS))
    refkeys = setdiff(refvalues, tablekeys)
    for key in refkeys
        table[key] = ""
    end
    return table     
end

function replacekeys(text::String, table::Dict)
    for (key, value) in pairs(table)
        if occursin(key, lowercase(text))
            return value
        end
    end
    return replace(text, "'"=>"")
end

function generatetables(description)
    tables = Dict()
    for (section, fields) in pairs(description)
        tables[section] = DataFrame(fill([], 8), [:Field, :Type, :Default, :Unit, :Pattern, :Implemented, :Description, :Concerns])
        for (field, lines) in fields
            columns = separatecolumns(lines)
            columns["Field"] = field
            push!(tables[section], columns)
        end
    end
    return tables
end

reftables = generatetables(description)
tablelist = let list = Vector{Pair{String, DataFrame}}(), df = DataFrame()
    for title in keys(SECTIONS_KEYS)
        table = reftables[title]
        push!(list, title=>table)
        df = vcat(df, table)
    end
    pushfirst!(list, "Full Table"=>df)
end


# Save tables to Excel Document
xlsxfile = joinpath(pwd(), "data\\pdp_units_data.xlsx")
XLSX.writetable(xlsxfile, tablelist; overwrite=true)

# Save Section keys
jsonfile = joinpath(pwd(), "data\\section_keys.json")
open(jsonfile, "w") do file
    keystring = JSON.json(SECTIONS_KEYS)
    write(file, keystring)
    close(file)
end