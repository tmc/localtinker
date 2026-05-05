# Exceptions

localtinker maps expected unsupported or invalid requests to SDK-visible user
errors. Unexpected local failures are returned as system errors.

## User Errors

Examples include:

- unsupported tensor forms;
- invalid sampling stop sequences;
- negative `topk_prompt_logprobs`;
- missing model, session, sampler, run, or checkpoint IDs;
- hosted-only operations not implemented locally.

## System Errors

System errors indicate local coordinator, checkpoint, MLX, tokenizer, model
cache, or filesystem failures.

## Authentication

localtinker does not validate hosted credentials. Set `TINKER_API_KEY` if the
SDK environment requires it, but the local server does not use its value for
authorization.
