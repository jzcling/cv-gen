from transformers import pipeline
import time
import asyncio
from math import floor
import re

class JDParser:
    def __init__(self):
        self.threshold_conf = 0.93

    async def classify(self, executor, job_title: str, text: str):
        start = time.time()

        lines = text.splitlines()
        lines = [line for line in lines if len(line) > 0]
        print(lines)
        
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(executor, self.classify_line, job_title, line) for line in lines]
        
        completed, _ = await asyncio.wait(tasks)
        result = [t.result() for t in completed]

        print(f'Classify Elapsed time: {time.time() - start:.3f}')
        
        return result

    def classify_line(self, job_title: str, text: str):
        start = time.time()
        
        keywords = self.keywords(text)
        description_type = self.description_type(text)
        classes = self.classes(job_title, description_type, text)

        result = {
            'text': text,
            'keywords': keywords,
            'classes': classes
        }

        print(f'Classify Line Elapsed time: {time.time() - start:.3f}')

        return result

    def keywords(self, text: str):
        start = time.time()

        keywords = ['quantum communications protocols', 'ci/cd', 'distributed systems', 
            'large scale storage', 'notifications', 'relational database', 
            'non relational database', 'concurrency', 'multithreading', 
            'synchronization', 'virtualization', 'load balancing', 'networking', 
            'massive data storage', 'security', 'monitoring', 'algorithms', 
            'command line', 'logging', 'test driven development', 'hardware in the loop', 
            'debugging', 'troubleshooting', 'failure analysis', 'design of experiment', 
            'ui/ux', 'threading', 'scrum', 'deep learning', 'computer vision', 
            'machine learning', 'artificial intelligence', 'etl', 'trading software', 
            'object oriented programming', 'oo programming', 'proof of concept', 'fintech', 
            'big data', 'containerization', 'infrastructure as code', 'key value store', 
            'sprint', 'payments', 'pair programming', 'ide extension', 
            'devops', 'logistics', '3d graphics', 'microservices']
        
        regex_keywords = ['UAT', 'Github', 'SQL', '[Nn]oSQL', 'Mongo(db|DB)', 'Cassandra', 
            'Python', 'Java(?![sS])', 'Go[,. ]', 'Java[sS]cript', 'Ruby', 'HTML', 'CSS', 'C\\+\\+', 
            'C#', 'C[,. ]' 'PHP', 'Swift', 'Kotlin', 'Android', '[Ii]OS', 'Scala', 'Rust', 
            'Perl', 'Matlab', 'R[,. ]', 'Flask', 'Django', 'Dash', 'Docker', 'Kubernetes', 
            '(Vue|React|Angular)\\.?(js)']
        
        regex_courses = ['(computer|software|electric|electrical|) engineering', 
            'computer science', 'information (systems|technology)',
            'mathematics', 'physics', 'statistics']
        
        cleaned_text = text.lower().replace("-", " ")

        matching_keywords = []
        for kw in keywords:
            if kw in cleaned_text:
                matching_keywords.append(kw)
        
        for kw in regex_keywords:
            match = re.search(kw, text)
            if match:
                matching_keywords.append(match.group(0))

        for course in regex_courses:
            match = re.search(course, cleaned_text)
            if match:
                matching_keywords.append(match.group(0))

        print(f'Keywords Elapsed time: {time.time() - start:.3f}')

        return matching_keywords

    def description_type(self, text: str):
        start = time.time()

        labels = ['ability and personal traits', 'skills and work experience', 'academic qualifications']

        cleaned_text = text.replace("\t", " ")
        classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")
        result = classifier(cleaned_text, labels)
        description_type = result['labels'][0]

        print(f'Desc Type Elapsed time: {time.time() - start:.3f}')

        return description_type

    def classes(self, job_title: str, description_type: str, text: str, multi_class: bool = True):
        start = time.time()

        labels = []
        if description_type == 'ability and personal traits':
            labels = ['adaptability', 'analytical', 'attention to detail', 'clean code', 
                'collaborate', 'communication of ideas', 
                'creativity', 'cross-functional collaboration', 'cross-geographical collaboration', 
                'diverse cultures', 'documentation', 'fast-paced environment', 
                'implement solutions', 'independent work', 'integrity', 
                'learning ability', 'open to learning', 'open to travel', 
                'prioritise deliverables', 'problem-solving ability', 
                'product management', 'resourcefulness', 'self-motivated', 
                'software design and app architecture', 'strategical', 
                'teaching ability', 'teamwork', 'trend awareness', 
                'work under pressure', 'written and verbal skills']
        elif description_type == 'skills and work experience':
            labels = ['agile development', 'automated testing', 'backend development', 
                'backend frameworks', 'best practices', 'clean code', 'cloud services', 
                'code review', 'collaboration', 'communication of ideas', 
                'communication to non-technical audience', 'conducting training', 
                'creating proposals', 'creativity', 
                'crisis management', 'cross functional collaboration', 
                'cross-geographical collaboration', 'customer facing products', 'customer support', 
                'data analytics', 'data processing', 'data science', 
                'data streaming', 'data structures', 'data visualization', 
                'data-driven development', 'database design', 'database management', 
                'deployment in production environment', 
                'design driven development', 'documentation', 'end-to-end development', 
                'enterprise web apps', 'error handling', 'evaluating tradeoffs', 
                'event-driven programming', 'finance industry', 'frontend development', 
                'frontend frameworks', 'functional requirements', 'gui development', 
                'hardware integration', 'highly available data systems', 'image processing', 
                'implementing solutions', 'independent work', 
                'integration with third-party software', 'internet services', 'mentoring', 
                'mobile development', 'on-premise deployments', 
                'operating systems', 'performance tuning', 
                'prioritising deliverables', 'problem-solving ability', 'product management', 
                'product roadmap', 'product-oriented', 'productivity tools', 
                'reactive programming', 'release management', 
                'restful api development', 'risk identification', 'scalable solutions', 
                'project scoping', 'session management', 
                'software design and app architecture', 'software development cycle', 
                'software maintenance', 'source control management', 'statistical methods', 
                'system availability', 'teamwork', 'technical support', 'test calibration', 
                'test validation', 'testing', 'trend awareness', 'user experience', 
                'written and verbal skills']
        elif description_type == 'academic qualifications':
            labels = ['bachelor\'s degree', 'diploma', 'master\'s degree', 'phd']

        cleaned_text = text.replace("\t", " ")

        classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")
        result = classifier(cleaned_text, labels, multi_class=multi_class)

        truncated_classes = {
            'sequence': '',
            'labels': [],
            'scores': []
        }
        truncated_classes['sequence'] = result['sequence']

        i = 0
        while result['scores'][i] >= self.threshold_conf:
            truncated_classes['labels'].append(result['labels'][i])
            truncated_classes['scores'].append(result['scores'][i])
            i += 1

        print(f'Classes Elapsed time: {time.time() - start:.3f}')

        return truncated_classes