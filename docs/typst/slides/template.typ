#import "@preview/definitely-not-isec-slides:1.0.1": *
#import "@preview/tableau-icons:0.331.0": *
#import "@preview/muchpdf:0.1.1": muchpdf
#import "@preview/booktabs:0.0.4": *

#let theme_color_primary_hm = rgb("fc5555")
#let theme_color_block = rgb("f4f6fb")
#let theme_color_footer = rgb("808080")

// Redefine the slide function to use custom logo in header (no institute name)
#let slide(
  title: auto,
  alignment: none,
  outlined: true,
  ..args,
) = touying-slide-wrapper(self => {
  let info = self.info + args.named()

  // Custom Header with logo only (no institute name)
  let header(self) = {
    let hdr = if title != auto { title } else { self.store.header }
    show heading: set text(size: 24pt, weight: "semibold")

    grid(
      columns: (self.page.margin.left, 1fr, auto, 0.5cm),
      block(), heading(level: 1, outlined: outlined, hdr), move(dy: -0.31cm, self.store.logo), block(),
    )
  }

  // Footer with page numbers and date
  let footer(self) = context {
    set block(height: 100%, width: 100%)
    set text(size: 15pt, fill: self.colors.footer)

    grid(
      columns: (self.page.margin.bottom - 1.68%, 1.3%, auto, 1cm),
      block(fill: self.colors.primary)[
        #set align(center + horizon)
        #set text(fill: white, size: 14pt)
        #utils.slide-counter.display()
      ],
      block(),
      block[
        #set align(left + horizon)
        #set text(size: 14pt)
        #info.at("footer", default: "")
      ],
      block(),
    )

    if self.store.progress-bar {
      place(bottom + left, float: true, move(dy: 1.05cm, components.progress-bar(
        height: 3pt,
        self.colors.primary,
        white,
      )))
    }
  }

  let self = utils.merge-dicts(self, config-page(
    header: header,
    footer: footer,
  ))

  set align(
    if alignment == none {
      self.store.default-alignment
    } else {
      alignment
    },
  )

  touying-slide(self: self, ..args)
})

// Default figure caption (title) styling for slides.
#show figure.caption: set text(size: 14pt)

// Override color-block to have rounded corners
#let color-block(
  title: [],
  icon: none,
  spacing: 0.78em,
  color: none,
  color-body: none,
  body,
) = [
  #touying-fn-wrapper((self: none) => [
    #show emph: it => {
      text(weight: "medium", fill: self.colors.primary, it.body)
    }

    #showybox(
      title-style: (
        color: white,
        sep-thickness: 0pt,
      ),
      frame: (
        radius: 8pt, // Rounded corners!
        thickness: 0pt,
        border-color: if color == none { self.colors.primary } else { color },
        title-color: if color == none { self.colors.primary } else { color },
        body-color: if color-body == none { self.colors.lite } else { color-body },
        inset: (x: 0.55em, y: 0.65em),
      ),
      above: spacing,
      below: spacing,
      title: if icon == none {
        align(horizon)[#strong(title)]
      } else {
        align(horizon)[
          #draw-icon(icon, height: 1.2em, baseline: 20%, fill: white) #h(0.2cm) #strong[#title]
        ]
      },
      body,
    )
  ])
]

// Parse the training step from filenames shaped like:
// `confusion_matrix_<step>_<hash>.png`.
#let conf-matrix-step(file) = {
  let base = file.split("/").last()
  let stem = base.replace(".png", "")
  let parts = stem.split("_")
  int(parts.at(2))
}

// Build ordered confusion-matrix frames from a directory manifest.
// `manifest` should be a JSON array of filenames (or a dict with `files`).
#let conf-matrix-frames-from-dir(dir, manifest: "frames.json") = {
  let raw = json(dir + "/" + manifest)
  let files = if type(raw) == dictionary { raw.at("files", default: ()) } else { raw }
  files.map(file => (step: conf-matrix-step(file), file: file)).sorted(key: it => it.step)
}

// Animated slide sequence for confusion matrices (Touying subslides).
// - `dir` is the directory containing the images (no trailing slash).
#let conf-matrix-sequence(
  dir,
  manifest: "frames.json",
  title: auto,
  alignment: none,
  outlined: true,
  width: 85%,
  caption: none,
  caption-style: (size: 14pt, weight: "medium"),
  show-step: true,
  step-prefix: [Step],
  step-style: (size: 12pt, fill: gray),
  ..args,
) = {
  let frames = conf-matrix-frames-from-dir(dir, manifest: manifest)
  slide(
    title: title,
    alignment: alignment,
    outlined: outlined,
    repeat: frames.len(),
    self => [
      #let frame = frames.at(self.subslide - 1)
      #align(center + horizon)[
        #figure(
          image(dir + "/" + frame.file, width: width),
          caption: if caption == none {
            none
          } else if show-step {
            [
              #text(
                ..caption-style,
              )[
                #caption #h(0.4em) #text(..step-style)[#step-prefix #frame.step]
              ]
            ]
          } else {
            caption
          },
        )
      ]
    ],
    ..args,
  )
}
