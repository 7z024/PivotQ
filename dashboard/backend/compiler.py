from .program import source_request, seal_program

def compile_project(project, plan=None):
    try:
        request = source_request({'task_id':project.task_id,'source':project.files.get('main.py','')})
        if plan is None:
            from .registry import validate_and_plan
            errors, plan = validate_and_plan(request)
            if errors:
                raise ValueError('; '.join(e['message'] for e in errors))
        program = seal_program(request, plan)
        return {'valid':True,'diagnostics':[],'program':program,'circuit':program['circuit'],
                'artifacts':{'execution_plan':plan.as_dict(),'source_sha256':program['source_sha256']}}
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        return {'valid':False,'diagnostics':[{'severity':'error','path':'main.py','message':str(error)}],'artifacts':{},'circuit':None}
