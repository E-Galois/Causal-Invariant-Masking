def answer_checking_common(args, ans, data, llm):
    t1 = data["text"]["answer"]
    t2 = ans["text"]
    prompt = f"Question: {{{data['text']['question']}}} True Answer: {{{t1}}} Is the following answer correct? Answer: {{{t2}}} Respond with 'no' if this is totally different from the True Answer. Respond with 'yes' when they are of the same meaning."
    ans = llm.generate(prompt, temp=0.1)
    return "yes" in ans.lower()


def answer_checking_okvqa(args, ans, data, llm):
    gts = set()
    for t1 in data["text"]["answer"]:
        if t1 in gts:
            continue
        gts.add(t1)
        t2 = ans["text"]
        prompt = f"Question: {{{data['text']['question']}}} True Answer: {{{t1}}} Is the following answer correct? Answer: {{{t2}}} Respond with 'no' if this is totally different from the True Answer. Respond with 'yes' when they are of the same meaning."
        llm_ans = llm.generate(prompt, temp=0.1)
        if "yes" in llm_ans.lower():
            return True
    return False
