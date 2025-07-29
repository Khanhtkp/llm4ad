from llm4ad.task.optimization.mo_tsp_construct import MOTSPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.nsga2 import NSGA2, NSGA2Profiler

if __name__ == '__main__':
    llm = HttpsApi(
        key='YOUR_API_KEY',
        model='gemini-2.5-flash',
        timeout=60
    )
    task = MOTSPEvaluation()
    method = NSGA2(
        llm=llm,
        profiler=NSGA2Profiler(log_dir='logs/eoh', log_style='complex'),
        evaluation=task,
        max_sample_nums=20,
        max_generations=10,
        pop_size=4,
        num_samplers=1,
        num_evaluators=1,
        debug_mode=True
    )
    method.run()