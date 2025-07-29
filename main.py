from llm4ad.task.optimization.tsp_construct import TSPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH, EoHProfiler

if __name__ == '__main__':
    llm = HttpsApi(
        key='YOUR_API_KEY',
        model='gemini-2.5-flash',
        timeout=60
    )
    task = TSPEvaluation()
    method = EoH(
        llm=llm,
        profiler=EoHProfiler(log_dir='logs/eoh', log_style='complex'),
        evaluation=task,
        max_sample_nums=20,
        max_generations=10,
        pop_size=4,
        num_samplers=1,
        num_evaluators=1,
        debug_mode=False
    )
    method.run()