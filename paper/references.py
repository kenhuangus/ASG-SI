"""APA 7 reference entries generated from verified metadata (arXiv/DataCite, Crossref, publisher pages).

arXiv metadata was retrieved from DataCite DOIs 10.48550/arXiv.<id> and cross-checked on arxiv.org/abs pages;
non-arXiv entries were checked against Crossref or the publisher's page.
"""
import json
import sys
from pathlib import Path

META = json.loads((Path(__file__).parent / "arxiv_metadata.json").read_text(encoding="utf-8"))
BY_ID = {m["id"]: m for m in META}

# key -> (arXiv id, optional venue note)
ARXIV = {
    "chen2025experience": ("2511.03773", None),
    "joshi2025swebenchcl": ("2507.00014", None),
    "li2025constraint": ("2512.20798", None),
    "luo2025lightning": ("2508.03680", None),
    "qian2025toolrl": ("2504.13958", None),
    "wu2025hidden": ("2509.21882", None),
    "wang2025skilllibrary": ("2512.17102", None),
    "wen2025rlvr": ("2506.14245", None),
    "yan2025memoryr1": ("2508.19828", None),
    "zhang2025landscape": ("2509.02547", None),
    "zhang2025agentrl": ("2510.04206", None),
    "zhou2025mem1": ("2506.15841", None),
    "su2025progressive": ("2512.07478", None),
    "lightman2023verify": ("2305.20050", None),
    "cobbe2021verifiers": ("2110.14168", None),
    "wang2023voyager": ("2305.16291", None),
    "chen2023skillit": ("2307.14430", "Advances in Neural Information Processing Systems 36"),
    "shridhar2021alfworld": ("2010.03768", "International Conference on Learning Representations", 2021),
    "shinn2023reflexion": ("2303.11366", None),
    "gou2023critic": ("2305.11738", None),
    "huang2023cannot": ("2310.01798", None),
    "schick2023toolformer": ("2302.04761", None),
    "cai2025smartsnap": ("2512.22322", None),
    "lin2025superintelliagent": ("2511.23436", None),
}

MANUAL = {
    "clopper1934": "Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial limits illustrated in the case of the binomial. *Biometrika, 26*(4), 404–413. https://doi.org/10.1093/biomet/26.4.404",
    "necula1997pcc": "Necula, G. C. (1997). Proof-carrying code. In *Proceedings of the 24th ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages* (pp. 106–119). https://doi.org/10.1145/263699.263712",
    "sha2001simplicity": "Sha, L. (2001). Using simplicity to control complexity. *IEEE Software, 18*(4), 20–28. https://doi.org/10.1109/MS.2001.936213",
    "torresarias2019intoto": "Torres-Arias, S., Afzali, H., Kuppusamy, T. K., Curtmola, R., & Cappos, J. (2019). in-toto: Providing farm-to-table guarantees for bits and bytes. In *Proceedings of the 28th USENIX Security Symposium* (pp. 1393–1410). https://www.usenix.org/conference/usenixsecurity19/presentation/torres-arias",
}


def initials(given):
    parts = [p for p in given.replace("-", " -").split() if p]
    out = []
    for p in parts:
        if p.startswith("-"):
            out[-1] = out[-1] + "-" + p[1] + "."
        elif p.endswith("."):
            out.append(p)
        else:
            out.append(p[0] + ".")
    return " ".join(out)


def author_list(authors):
    names = [f"{fam}, {initials(giv)}" if giv else fam for fam, giv in authors]
    if len(names) == 1:
        return names[0]
    if len(names) <= 20:
        return ", ".join(names[:-1]) + ", & " + names[-1]
    return ", ".join(names[:19]) + ", … " + names[-1]


def arxiv_entry(key):
    aid, venue, *year = ARXIV[key]
    m = BY_ID[aid]
    title = m["title"].rstrip(".")
    base = f"{author_list(m['authors'])} ({year[0] if year else m['year']}). {title}."
    if venue:
        return f"{base} *{venue}*. https://arxiv.org/abs/{aid}"
    return f"{base} *arXiv preprint arXiv:{aid}*. https://arxiv.org/abs/{aid}"


def all_entries():
    entries = {k: arxiv_entry(k) for k in ARXIV}
    entries.update(MANUAL)
    return dict(sorted(entries.items(), key=lambda kv: kv[1].lower()))


if __name__ == "__main__":
    for k, v in all_entries().items():
        print(k, "=>", v)
