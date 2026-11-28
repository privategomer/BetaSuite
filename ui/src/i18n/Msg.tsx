import type { CatalogKey } from './catalog';
import { t, type DynamicKey, type MessageValues } from './format';

interface MsgProps {
  id: CatalogKey | DynamicKey;
  values?: MessageValues;
}

// A catalog message as a React node.
export function Msg( { id, values }: MsgProps ) {
  return <>{t( id, values )}</>;
}
