from fastapi import APIRouter, HTTPException
from db import get_async_connection
from model import embed
import asyncio
from pydantic import BaseModel

router = APIRouter()

class UpdateEmbeddingRequest(BaseModel):
    productName: str
    childestCategoryName: str = ""
    description: str = ""

@router.put("/update-embedding/{product_id}")
async def update_product_embedding(
    product_id: int,
    data: UpdateEmbeddingRequest
):
    """
    Updates both Embedding and TSV for a product.
    
    Embedding: ProductName + Childest Category + Description
    TSV: ProductName + Childest Category (for word-by-word search)
    
    ASP.NET should call this after creating/updating a product.
    """
    conn = await get_async_connection()
    try:
        # Build embedding text: ProductName + Childest Category + Description
        embed_text = f"{data.productName} {data.childestCategoryName} {data.description}".strip()
        
        # Build TSV text: ProductName + Childest Category (no description)
        tsv_text = f"{data.productName} {data.childestCategoryName}".strip()
        
        # Generate embedding
        vector = await asyncio.to_thread(embed, embed_text)
        vector_str = "[" + ", ".join(map(str, vector)) + "]"
        
        # Update BOTH Embedding and TSV (with quotes for case-sensitive table/column names)
        result = await conn.execute(
            """
            UPDATE "Products"
            SET "Embedding" = $2::vector,
                "TSV" = to_tsvector('simple', unaccent($3))
            WHERE "ProductId" = $1
            """,
            product_id, vector_str, tsv_text
        )
        
        if result == "UPDATE 0":
            raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
        
        return {
            "status": "ok", 
            "productId": product_id,
            "embeddedText": embed_text,
            "tsvText": tsv_text
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await conn.close()