from typing import Optional

from pydantic import BaseModel, Field


class MunicipalReport(BaseModel):
    is_waste_present: bool = Field(description="True if urban waste/garbage is visible.")
    waste_type: str = Field(description="E.g., Plastic, Construction, Organic Mixed.")
    # TODO: specify if lower is better or worse
    severity_score: int = Field(description="Scale of 1-10 based on volume and public hazard.", ge=0, le=10)
    recommendded_equipment: list[str] = Field(description="List of tools needed (e.g., ['Broom', 'JCB]).")
    action_required: str = Field(description="A 1-sentence alert for the municipal officer.")


class DuplicateCheck(BaseModel):
    is_same: bool = Field(description="True if waste is detected on road, else false")


class ImageExif(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
