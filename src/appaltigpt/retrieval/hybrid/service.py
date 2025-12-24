import asyncio
from typing import List, Dict, Any, Union
from appaltigpt.retrieval.ports.retriever_port import RetrieverPort
from appaltigpt.retrieval.ports.keyword_search_port import KeywordSearchPort

class HybridRetrieverService:
    """
    Servizio di ricerca Ibrida generico.
    Può fondere i risultati di un numero arbitrario di retriever (Semantic, HyDE, Keyword, ecc.)
    usando Reciprocal Rank Fusion (RRF).
    """
    
    def __init__(
        self,
        retrievers: List[Union[RetrieverPort, KeywordSearchPort]],
        rrf_k: int = 60
    ):
        self.retrievers = retrievers
        self.rrf_k = rrf_k

    async def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:

        tasks = []
        for retriever in self.retrievers:
            if hasattr(retriever, 'retrieve'):
                tasks.append(retriever.retrieve(query, limit=limit))
            elif hasattr(retriever, 'search'):
                tasks.append(retriever.search(query, limit=limit))
            else:

                continue
        

        results_lists = await asyncio.gather(*tasks)
        

        fused_results = self._reciprocal_rank_fusion(results_lists, limit)
        
        return fused_results

    def _reciprocal_rank_fusion(
        self, 
        results_lists: List[List[Dict[str, Any]]], 
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Combina N liste di risultati usando RRF.
        """
        scores: Dict[str, float] = {}
        content_map: Dict[str, Dict[str, Any]] = {}

        for doc_list in results_lists:
            for rank, item in enumerate(doc_list):
                doc_id = str(item['id'])
                if doc_id not in content_map:
                    content_map[doc_id] = item
                


                scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank + 1))


        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        

        final_results = []
        for doc_id in sorted_ids[:limit]:
            item = content_map[doc_id]
            item_copy = item.copy()
            item_copy['hybrid_score'] = scores[doc_id]
            final_results.append(item_copy)
            
        return final_results
