from src.pipeline import qa_with_retrieval

def run(img, q):
    out = qa_with_retrieval(img, q)
    print("Q:", q)
    print("A:", out["answer"])
    print("Title:", out["evidence_title"])
    print("Reflect:", out["reflection"])
    print("Ev:", (out["evidence"] or "")[:220].replace("\n"," "))
    print("---")

IMG1 = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
Q1 = "What animal is this"

IMG2 = "https://upload.wikimedia.org/wikipedia/commons/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg"
Q2 = "When was this tower built"

run(IMG1, Q1)
run(IMG2, Q2)
