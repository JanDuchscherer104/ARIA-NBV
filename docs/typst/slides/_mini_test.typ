#import "template.typ": *
#import "../shared/macros.typ": *

#show: definitely-not-isec-theme.with(
  aspect-ratio: "16-9",
  slide-alignment: top,
  progress-bar: false,
  institute: [Test],
  logo: [],
  config-info(
    title: [Test],
    subtitle: [],
    authors: [],
    extra: [],
    footer: [],
    download-qr: "",
  ),
  config-common(handout: false),
  config-colors(primary: theme_color_primary_hm, lite: theme_color_block),
)

#slide(title: [Test])[
  - Modules: #code-inline[transform_points_screen] + #code-inline[_encode_semidense_projection_features].

  #code-figure(caption: [Snippet])[
    ```python
    def f():
        return 1
    ```
  ]
]
