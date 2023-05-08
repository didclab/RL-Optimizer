import json
import unittest
import ods_helper


class OdsHelperTest(unittest.TestCase):

    def test_send_application_params(self):
        resp = ods_helper.send_application_params_tuple(
            3, 10, 1, "jgoldverg@gmail.com-mac", 0)
        assert resp.status_code == 200

    def test_query_if_job_done(self):
        resp, batch_info = ods_helper.query_if_job_done(12267)
        print(resp)
        bi = batch_info
        print(bi)
        assert resp == True

    def test_submit_transfer_job_request(self):
        resp, batch_info = ods_helper.query_if_job_done(12267)
        transferRequest = ods_helper.transform_batch_info_json_to_transfer_request(
            batch_info)
        assert transferRequest.ownerId == "jgoldverg@gmail.com"

    def test_submit_transfer_request(self):
        resp, batch_info = ods_helper.query_if_job_done(12274)
        transferRequest = ods_helper.transform_batch_info_json_to_transfer_request(
            batch_info)
        assert transferRequest.ownerId == "jgoldverg@gmail.com"
        print(transferRequest.toJSON())
        transfer_response = ods_helper.submit_transfer_request(batch_info, '')
        print(transfer_response)
        print(transfer_response.json())
        assert transfer_response.status_code == 200

    def test_submit_transfer_request_ddpg(self):
        resp, batch_info = ods_helper.query_if_job_done(12274)
        transferRequest = ods_helper.transform_batch_info_json_to_transfer_request(
            batch_info)
        assert transferRequest.ownerId == "jgoldverg@gmail.com"
        print(transferRequest.toJSON())
        transfer_response = ods_helper.submit_transfer_request(
            batch_info, 'DDPG')
        print(transfer_response)
        print(transfer_response.json())
        assert transfer_response.status_code == 200

    def test_submit_transfer_request_bdq(self):
        """
        Not ready yet
        """
        resp, batch_info = ods_helper.query_if_job_done(12274)
        transferRequest = ods_helper.transform_batch_info_json_to_transfer_request(
            batch_info)
        assert transferRequest.ownerId == "jgoldverg@gmail.com"
        print(transferRequest.toJSON())
        transfer_response = ods_helper.submit_transfer_request(
            batch_info, 'BDQ')
        print(transfer_response)
        print(transfer_response.json())
        assert transfer_response.status_code == 200
