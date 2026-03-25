## Context
- Streamlit depth stage crashed after simplifying `CandidateDepthRenderer`, showing warning `target` during depth rendering.
- Cause: runtime type checks used `Efm3dDepthRendererConfig.target` / `Pytorch3DDepthRendererConfig.target`. In pydantic v2 class attributes for fields are masked, so accessing `.target` on the config class raises `AttributeError('target')`, aborting before rendering.

## Fix
- Compare against renderer classes instead of config `.target` fields.
- Updated `candidate_depth_renderer.py` to import `Efm3dDepthRenderer` and `Pytorch3DDepthRenderer`, use `isinstance(..., Pytorch3DDepthRenderer)` for mesh selection, and raise a clear `TypeError` for unsupported backends.

## Notes / Next Steps
- Depth rendering should proceed again in the Streamlit app; verify via Candidate Renders tab.
- Consider adding a small unit test covering renderer type checks to prevent regression.
