# Pro-Scout web

A static Next.js site. The tactics board runs the Phase 2 passing-network models in the visitor's browser, so
there is no server.

```powershell
# from the repository root: the data the site serves (Python side)
python -m sim export-web                 # -> web/public/data/
python -m sim rankings --workers 8       # best replacements for every XI (~1 h, resumable)

cd web
npm install
npm test                                 # TypeScript networks vs PyTorch (needs test/golden.json:
                                         #   python -m sim export-web --golden web/test/golden.json)
npm run dev                              # http://localhost:3000
npm run build                            # static site in out/
```

- `lib/model.ts`: DeepSets and the GINE GNN, inference only, mirroring `models/gnn/net.py`.
- `lib/sim.ts`: swaps, mirroring `sim/swap.py` (`Simulator.swap_many`) and `sim/profiles.py`.
- `lib/worker.ts`: runs the networks in a Web Worker so the page stays responsive.
- `components/`: the board (`Pitch`), choosing a replacement (`Picker`), the result (`Result`).

Deploying on Vercel: import the GitHub repository, set **Root Directory** to `web`; the framework (Next.js) and
build command are detected. The site is exported as static files (`output: "export"`).
