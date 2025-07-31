from llm4ad.task.optimization.mo_tsp_construct import MOTSPEvaluation
from llm4ad.tools.llm.llm_api_gemini import GeminiApi
from llm4ad.method.momcts import MOMCTS_AHD, MAProfiler

if __name__ == '__main__':
    llm = GeminiApi(
        key='YOUR_API_KEY',
        model='gemini-2.5-flash',
        timeout=60
    )
    task = MOTSPEvaluation()
    method = MOMCTS_AHD(
        llm=llm,
        profiler=MAProfiler(log_dir='logs/momcts', log_style='complex'),
        evaluation=task,
        max_sample_nums=20,
        max_generations=10,
        pop_size=4,
        num_samplers=1,
        num_evaluators=1,
        debug_mode=True
    )
    method.run()