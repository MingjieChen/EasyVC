#!/bin/bash



splits=$1
dataset=$2

if [ ! -e ling_encoder/contentvec_100/contentvec_100_model.pt ]; then
    cd ling_encoder/contentvec_100
    wget "https://public.boxcloud.com/d/1/b1\!aZwF89Eh4jXgxr7yGU4SQ7YuIF7zrYJ_QyySriX1CBnI5Cg0oI_5mKHTcakLamdA-XyLSzH5Natnk-mDc-ea7crePAVToem5nSlaGtTBWaT4H8QZ5FsJv82YS7jyBjzeZ-20V34qmZKLgAWckoeA8t5utnBTrFP7URXoPcGST5Kn3pnMGhD2sgClyELFGB0PedSRegfxq9RBNqmx7KAgdxerreUI_EafNH29SxqVC2H2c0aioUu8Qmp_nObVKRdysGBF-rvcfx08lAoLoGaF_dw_KxDbvDEi6GLmIDBBAWE55_5IOqFvE9PZxw8tXHT4fXWiyBpoTgMkN7jOiAd-2BeYGrRome9gaPnwG-GegeDcVAqseMqlOqMXaFTQsEol2fNeFt427l9-9PBtfyiZlW9ru-APHtDHIErIRSgIaMViJA368lssKN7vlXJaUJ4qUNAik37hP0HeTfuQbppxDLhRfPVRTT8tZtR40PkX2UGz6Ev8_b_R7jfk6CkWsQ-U3znUJmudMM_OLcZPS7ZsZSU1M4_rzo2sCg4p80KYbpNaAkKNblODEH6WQnYNC-FDPwHH-6FZMrs8DnA-TcHuCNWBLAl9Oe4vReHYHSSLoxHUyVcqe_ox5klCinuxgG1cY7AOzXfq8imFvvsyJrwa3bsXs0Y_Ql4pUcvekGQKslbDqbpb7jkXOnqvFEvUzt71l6UWA64dCFYFNYo0HQWtaj6VfLxCpj9dL60d5fDmjTyRii96qdZYUU04kPmNgqrQv7hufMod0QkuKkpjIewaaGj3cyAJxXiY3Tv5QhXOvNizxm5yJOzCDUJgiWWWMSISSS8578azEE_Fk2N43IOIfSqG7rQBXyZzxlKlDLuFtA7OBqAiLlyT55ZSUK8GrnCrn_BDu2UBUC7laWDK_SpdjsJcxnWSWzVIPI8NVysLHd1ejtAdGRkohATdf6tzlJaQDlWkj1D3p3wyq1q4YHd-KK_NVbTwLOj0Tz41O_iT5WMEIhbw-aJXhkFvHMdVzEVBFfVCoMWcrbEW7M7ix0bOt5NWB7A5dPFNqX_PrEG3VRIuWhbfSMwJPlv1f1Fck9ict8Fs6o_c0o8lNe7LROf1P-7pBPMWpgx-OgEhnRKYUkxkraXneunNr7V-yjjI1Ham99JpK02yAUATf3pEKjCJaUIMqmawYbv_nOl6rnjwwyTB3_t5O0h7PoNqJmVqeyzFpGp-gDx4ZZPkFsOXHQ6s8A8nRmTrTL3B_tMFMdvaqt0lv2KECstUO0pbk2yeeejIIhtjWVBq-L5JnTocz4HmhCnfwqguqbvGC8bOsdu5zWC5wvqgQGN7/download"
    ln -s checkpoint_best_legacy_100.pt contentvec_100_model.pt
    cd ../../
    echo "done!"
fi

for split in $splits ; do
    
    echo "[vqwav2vec feature extraction]: $split for libritts"
    python3 ling_encoder/contentvec_100/contentvec_100_feature_extract.py \
        --vqwav2vec_ckpt ling_encoder/contentvec_100/contentvec_100_model.pt \
        --metadata data/$dataset/metadata.csv \
        --dump_dir dump/$dataset \
        --split $split \
        --max_workers 20
done        

    
