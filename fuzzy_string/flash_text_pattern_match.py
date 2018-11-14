start_time = datetime.datetime.now()

conn = get_connection()

ids = list(map(lambda x:x[0], get_table_ids()))

def update_drugs(ids):
    drugs = list(map(lambda x: x[0], get_drugs()))
    keyword_processor = KeywordProcessor(case_sensitive=False)
    for d in drugs:
        keyword_processor.add_keyword((' '+d.strip()+' ').encode('utf-8'))
        keyword_processor.add_keyword(('-'+d.strip()).encode('utf-8'))
        keyword_processor.add_keyword((d.strip()+'-').encode('utf-8'))
        
    update_list = []
    sentences = get_sentences_to_search(ids)
    for sen in sentences:
        id = sen[0]

        found = []
        found = keyword_processor.extract_keywords(sen[1].encode('utf-8').strip().lower())
        if len(found)>0:
            _found = list(map(lambda x: x.decode('utf-8'), found))
            print(_found)
            print(sen[1].encode('utf-8').strip().lower())
            update_list.append({'id':id, 'drug':' '+' , '.join(_found)+' '})

    if len(update_list)>0:
        print('update')
        print(update_list)
        copy_into_table('drug', update_list)
    
    update_list_m = []
    sentences_m = get_sentences_to_search_m(ids)
    for sen in sentences_m:
        id = sen[0]

        found = []
        found = keyword_processor.extract_keywords(sen[1].encode('utf-8').strip().lower())
        if len(found)>0:
            _found = list(map(lambda x: x.decode('utf-8'), found))
            print(_found)
            print(sen[1].encode('utf-8').strip().lower())
            update_list_m.append({'id':id, 'drug':' '+' , '.join(_found)+' '})

    if len(update_list_m)>0:
        print('update m')
        copy_into_table_m('drug', update_list_m)
        
def update_llts(ids):
    side_effects = list(map(lambda x: x[0], get_side_effects()))
    keyword_processor = KeywordProcessor(case_sensitive=False)
    for side in side_effects:
        keyword_processor.add_keyword((' '+side.strip()+' ').encode('utf-8'))
        keyword_processor.add_keyword(('-'+side.strip()).encode('utf-8'))
        keyword_processor.add_keyword((side.strip()+'-').encode('utf-8'))
    
    update_list = []
    sentences = get_sentences_to_search(ids)
    for sen in sentences:
        id = sen[0]

        found = []
        found = keyword_processor.extract_keywords(sen[1].encode('utf-8').strip().lower())
        if len(found)>0:
            _found = list(map(lambda x: x.decode('utf-8'), found))
            print(_found)
            print(sen[1].encode('utf-8').strip().lower())
            update_list.append({'id':id, 'adverse_effect':' '+' , '.join(_found)+' '})

    if len(update_list)>0:
        print('update')
        print(update_list)
        copy_into_table('ad', update_list)
    
    update_list_m = []
    sentences_m = get_sentences_to_search_m(ids)
    for sen in sentences_m:
        id = sen[0]

        found = []
        found = keyword_processor.extract_keywords(sen[1].encode('utf-8').strip().lower())
        if len(found)>0:
            _found = list(map(lambda x: x.decode('utf-8'), found))
            print(_found)
            print(sen[1].encode('utf-8').strip().lower())
            update_list_m.append({'id':id, 'adverse_effect':' '+' , '.join(_found)+' '})

    if len(update_list_m)>0:
        print('update m')
        copy_into_table_m('ad', update_list_m)

for i in range(1000, len(ids)+1000, 1000):
    print(str(i-1000)+':'+str(i))
    print('llts : ' +str(update_llts(ids[i-1000:i])))
    print('drugs:' + update_drugs(ids[i-1000:i]))
    
print(start_time - datetime.datetime.now())