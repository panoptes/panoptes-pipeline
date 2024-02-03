from typing import Tuple

from google.cloud import firestore
from panoptes.data.observations import ObservationPathInfo


def get_firestore_refs(
        bucket_path: str,
        unit_collection: str = 'units',
        observation_collection: str = 'observations',
        image_collection: str = 'images',
        firestore_db: firestore.Client = None
) -> Tuple[firestore.DocumentReference, firestore.DocumentReference, firestore.DocumentReference]:
    """Gets the firestore image document reference"""
    firestore_db = firestore_db or firestore.Client()

    path_info = ObservationPathInfo(path=bucket_path)
    sequence_id = path_info.sequence_id
    image_id = path_info.image_id

    print(f'Getting firestore document for image {path_info.get_full_id()}')
    unit_collection_ref = firestore_db.collection(unit_collection)
    unit_doc_ref = unit_collection_ref.document(f'{path_info.unit_id}')
    seq_doc_ref = unit_doc_ref.collection(observation_collection).document(sequence_id)
    image_doc_ref = seq_doc_ref.collection(image_collection).document(image_id)

    return unit_doc_ref, seq_doc_ref, image_doc_ref
