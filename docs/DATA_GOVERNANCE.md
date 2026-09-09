# Data governance

Every dataset must have a versioned dataset card identifying origin, ownership or licence, collection method, intended use, prohibited use, transformations, filters, known limitations, privacy review, and content hash.

## Non-negotiable rules

- Do not commit personal, customer, confidential, restricted, or unlawfully obtained data.
- Do not assume public availability grants permission for model training.
- Keep training, validation, and hidden evaluation sets separate and contamination-checked.
- Record synthetic-data generator, model/version, prompt method, sampling settings, and review process.
- Preserve opt-out, deletion, and correction mechanisms where the source or applicable policy requires them.
- Quarantine uncertain data until a human completes licence and privacy review.

Dataset changes create a new immutable version and require evaluation against the previous baseline. Hashes demonstrate identity; they do not demonstrate legality, quality, or consent.
