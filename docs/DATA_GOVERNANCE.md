# Data governance

Every dataset must have a versioned dataset card identifying origin, ownership or licence, collection method, intended use, prohibited use, transformations, filters, known limitations, privacy review, and content hash.

Internet discovery produces quarantined source candidates by default. The research agent cannot approve its own discoveries for training. Admission requires an explicit intended use and independent rights/privacy review through the source-candidate contract.

## Non-negotiable rules

- Do not commit personal, customer, confidential, restricted, or unlawfully obtained data.
- Do not assume public availability grants permission for model training.
- Keep training, validation, and hidden evaluation sets separate and contamination-checked.
- Record synthetic-data generator, model/version, prompt method, sampling settings, and review process.
- Preserve opt-out, deletion, and correction mechanisms where the source or applicable policy requires them.
- Quarantine uncertain data until a human completes licence and privacy review.

Dataset changes create a new immutable version and require evaluation against the previous baseline. Hashes demonstrate identity; they do not demonstrate legality, quality, or consent.

## Multi-package held-out exclusion registries: track both directions, per package

Any project that builds up a dataset across multiple sequential
"packages" or extraction rounds, and shares a held-out-exclusion registry
across them to stop a later package from re-using an id another package
already spent as held-out, should apply two design rules learned from a
real contamination incident (see
[issue #11](https://github.com/Rook-CodeVolt/CodeVolt-Model-Development-Framework/issues/11)
for the full case study and evidence).

**Track both directions symmetrically.** It is not enough to track "used
as held-out by any package" and exclude those ids from later packages'
candidate pools — that only stops a later package from re-using an
earlier package's held-out ids as training data. It misses the reverse and
equally serious direction: an id already used as **train** data by an
earlier package must never later be selected as a **held-out** example by
a later package, because the model may already have seen that item during
the earlier package's training, which invalidates any "held-out" claim
about it in the later package's evaluation. A registry that only reasons
about "the held-out side" will not catch this. The incident behind this
guidance found 67 ids from one package's train split were also present in
a later package's held-out split, undetected until a general train/held-out
contamination check was run on every extraction rather than only checked
in the commonly-reviewed direction.

**Store registry state per package, not as one flat union.** A registry
that stores only a single, ever-growing flat set of excluded ids has no
way to represent "this package's contribution has been superseded." If a
package is re-split to remove contamination, its old ids remain in the
flat union forever, and the registry keeps flagging against data that no
longer exists in any current package — a false-positive that specifically
punishes the fix. Attribute each id to the package that registered it
(e.g. `package_held_out_ids` and `package_train_ids` keyed by package,
with any flat/global view derived from those rather than stored as its own
source of truth) so that a package's own re-registration supersedes only
its own prior contribution, leaving other packages' registrations intact.

Together these give a registry that (a) catches contamination in both
directions and (b) stays correct — rather than accumulating stale
false positives — across repeated corrections to any one package.
