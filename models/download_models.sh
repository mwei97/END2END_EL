if [[ ! -f entity_token_ids_128.t7 ]]; then
    wget http://dl.fbaipublicfiles.com/elq/entity_token_ids_128.t7
fi

if [[ ! -f all_entities_large.t7 ]]; then
    wget http://dl.fbaipublicfiles.com/BLINK/all_entities_large.t7
fi

if [[ ! -f faiss_hnsw_index.pkl ]]; then
    wget http://dl.fbaipublicfiles.com/elq/faiss_hnsw_index.pkl
fi

if [[ ! -f elq_wiki_large.bin ]]; then
    wget http://dl.fbaipublicfiles.com/elq/elq_wiki_large.bin
fi
