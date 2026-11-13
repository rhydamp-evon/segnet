from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


class AnalyzeRequest(BaseModel):
    analysis_id: UUID
    source_image_url: HttpUrl
    platform_id: str
    project_id: str
    content_id: str
    content_type: str


class RegionResponse(BaseModel):
    region_id: str
    category_code: str
    label: str
    level: Literal["structure", "surface", "fixture", "object"]
    parent_id: str | None = None
    confidence: float | None = None
    bbox: list[int] = Field(default_factory=list)
    overlay: dict[str, Any] = Field(default_factory=dict)
    mask_path: str


class AnalyzeResponse(BaseModel):
    source_image_hash: str
    model_version: str
    regions: list[RegionResponse]


class EditRegion(BaseModel):
    region_id: str
    category_code: str
    level: Literal["structure", "surface", "fixture", "object"]
    parent_id: str | None = None
    bbox: list[int] = Field(default_factory=list)
    mask_path: str


class EditRequest(BaseModel):
    job_id: UUID
    analysis_id: UUID
    source_image_url: HttpUrl
    source_image_hash: str
    remove_region_ids: list[str] = Field(default_factory=list)
    blur_region_ids: list[str] = Field(default_factory=list)
    blur_strength: int = Field(default=25, ge=1, le=100)
    refine_masks: bool = False
    regions: list[EditRegion]


class EditResponse(BaseModel):
    result_image_url: HttpUrl
    model_version: str
    mask_refiner_version: str | None = None
    refined_regions: int = Field(default=0, ge=0)


class DemoAnalyzeResponse(BaseModel):
    analysis_id: UUID
    prompts: list[str]
    model_variant: Literal["nano", "small", "sam3"]
    source_image_url: str
    source_image_hash: str
    model_version: str
    inference_ms: int = Field(ge=0)
    mask_refiner_version: str | None = None
    refined_regions: int = Field(default=0, ge=0)
    regions: list[RegionResponse]


class DemoEditRequest(BaseModel):
    analysis_id: UUID
    source_image_hash: str
    remove_region_ids: list[str] = Field(default_factory=list)
    blur_region_ids: list[str] = Field(default_factory=list)
    blur_strength: int = Field(default=25, ge=1, le=100)
    refine_masks: bool = False
    regions: list[EditRegion]


class DemoEditResponse(BaseModel):
    result_image_url: str
    model_version: str
    mask_refiner_version: str | None = None
    refined_regions: int = Field(default=0, ge=0)
