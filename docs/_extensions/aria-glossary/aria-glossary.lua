local function extension_dir()
  local source = debug.getinfo(1, "S").source or ""
  source = source:gsub("^@", "")
  source = source:gsub("\\", "/")
  return source:match("^(.*)/[^/]*$") or "."
end

local terms = dofile(extension_dir() .. "/glossary_terms.generated.lua")
local notation = dofile(extension_dir() .. "/notation.generated.lua")

local function stringify_arg(args, index)
  local value = args[index]
  if value == nil then
    return nil
  end
  local text = pandoc.utils.stringify(value):gsub("^%s+", ""):gsub("%s+$", "")
  if text == "" then
    return nil
  end
  return text
end

local function warn(message)
  if quarto and quarto.log and quarto.log.warning then
    quarto.log.warning(message)
  else
    io.stderr:write("WARNING: " .. message .. "\n")
  end
end

local function split_path(path)
  local segments = {}
  path = (path or ""):gsub("\\", "/")
  for segment in path:gmatch("[^/]+") do
    if segment ~= "." and segment ~= "" then
      table.insert(segments, segment)
    end
  end
  return segments
end

local function dirname(path)
  path = (path or ""):gsub("\\", "/")
  local dir = path:match("^(.*)/[^/]*$")
  return dir or ""
end

local function relative_url(from_dir, to_path)
  local from = split_path(from_dir)
  local to = split_path(to_path)
  local common = 0
  while common < #from and common < #to and from[common + 1] == to[common + 1] do
    common = common + 1
  end

  local parts = {}
  for _ = common + 1, #from do
    table.insert(parts, "..")
  end
  for index = common + 1, #to do
    table.insert(parts, to[index])
  end
  if #parts == 0 then
    return "."
  end
  return table.concat(parts, "/")
end

local function current_source_dir()
  if quarto and quarto.doc and quarto.doc.input_file and quarto.project and quarto.project.directory then
    local input = tostring(quarto.doc.input_file):gsub("\\", "/")
    local project = tostring(quarto.project.directory):gsub("\\", "/"):gsub("/$", "")
    local relative = input
    if relative:sub(1, #project + 1) == project .. "/" then
      relative = relative:sub(#project + 2)
    end
    return dirname(relative)
  end
  return ""
end

local function glossary_href(term)
  return relative_url(current_source_dir(), "contents/glossary.html") .. "#" .. term.anchor
end

local function term_text(term, full)
  if full and term.short ~= term.label then
    return term.label .. " (" .. term.short .. ")"
  end
  if full then
    return term.label
  end
  return term.short
end

local function render_term(args, full)
  local term_id = stringify_arg(args, 1)
  if term_id == nil then
    warn("glossary shortcode missing term id")
    return pandoc.Str("??gls:missing??")
  end

  local term = terms[term_id]
  if term == nil then
    warn("unknown glossary term id: " .. term_id)
    return pandoc.Str("??gls:" .. term_id .. "??")
  end

  local text = term_text(term, full)
  if quarto and quarto.doc and quarto.doc.isFormat and quarto.doc.isFormat("html") then
    local attr = pandoc.Attr("", { "glossary-ref" }, { ["data-glossary-id"] = term_id })
    return pandoc.Link({ pandoc.Str(text) }, glossary_href(term), "", attr)
  end
  return pandoc.Str(text)
end

local function render_notation(args, group, display)
  local notation_id = stringify_arg(args, 1)
  if notation_id == nil then
    warn(group .. " shortcode missing notation id")
    return pandoc.Str("??" .. group .. ":missing??")
  end

  local group_entries = notation[group] or {}
  local entry = group_entries[notation_id]
  if entry == nil then
    warn("unknown " .. group .. " notation id: " .. notation_id)
    return pandoc.Str("??" .. group .. ":" .. notation_id .. "??")
  end

  local math_type = "InlineMath"
  if display then
    math_type = "DisplayMath"
  end
  return pandoc.Math(math_type, entry.tex)
end

return {
  gls = function(args)
    return render_term(args, false)
  end,
  glsfull = function(args)
    return render_term(args, true)
  end,
  sym = function(args)
    return render_notation(args, "symbols", false)
  end,
  eq = function(args)
    return render_notation(args, "equations", true)
  end,
}
