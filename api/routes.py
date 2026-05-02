from fastapi import APIRouter, Depends
from pydantic import BaseModel
from api.auth import verify_api_key
from generators.blog import generate_blog
from generators.social import generate_social
from generators.email import generate_email
from generators.ad_copy import generate_ad
from generators.product_desc import generate_product_desc

router = APIRouter(prefix="/api/v1", tags=["content"])

class BlogRequest(BaseModel):
    topic: str
    brand_voice: str = "professional"
    keywords: list = []
    word_count: int = 800
    outline_only: bool = False

class SocialRequest(BaseModel):
    topic: str
    platform: str = "linkedin"
    brand_voice: str = "professional"
    count: int = 1

class EmailRequest(BaseModel):
    topic: str
    email_type: str = "promotional"
    brand_voice: str = "professional"
    cta: str = "Learn More"

class AdRequest(BaseModel):
    product: str
    ad_format: str = "meta_feed"
    brand_voice: str = "bold"
    cta: str = "Shop Now"

class ProductDescRequest(BaseModel):
    product_name: str
    features: list = []
    brand_voice: str = "professional"
    keywords: list = []
    platform: str = "ecommerce"

@router.post("/blog", dependencies=[Depends(verify_api_key)])
async def blog(req: BlogRequest):
    return generate_blog(req.topic, req.brand_voice, req.keywords, req.word_count, req.outline_only)

@router.post("/social", dependencies=[Depends(verify_api_key)])
async def social(req: SocialRequest):
    return generate_social(req.topic, req.platform, req.brand_voice, req.count)

@router.post("/email", dependencies=[Depends(verify_api_key)])
async def email(req: EmailRequest):
    return generate_email(req.topic, req.email_type, req.brand_voice, req.cta)

@router.post("/ad", dependencies=[Depends(verify_api_key)])
async def ad(req: AdRequest):
    return generate_ad(req.product, req.ad_format, req.brand_voice, req.cta)

@router.post("/product-description", dependencies=[Depends(verify_api_key)])
async def product_desc(req: ProductDescRequest):
    return generate_product_desc(req.product_name, req.features, req.brand_voice, req.keywords, req.platform)
